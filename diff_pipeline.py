import os
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict

# ==========================
# KONFIGURACJA DOMYŚLNA
# ==========================
EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg")

OUT_DIR_DEFAULT = "WYNIKI_DIFF_PIPELINE"

# Dopasowanie wzorca
SEARCH_SIZE = (128, 128)     # miniatury do MSE

# Parametry różnic
BLUR_K = 5
DIFF_THRESH = 45
ERODE_ITER = 2
DILATE_ITER = 4
KERNEL_SIZE = 3

# Usuwanie dołu (pasek / bloczki)
BOTTOM_CUT_FRAC = 0.93  # od tego miejsca w dół maska zerowana

# Filtrowanie blobów (jak w Twojej aplikacji)
MIN_AREA = 200
MAX_AREA = 15000
MIN_FILL = 0.15
MIN_AR, MAX_AR = 0.4, 4.0
BORDER_MARGIN = 5


# ==========================
# POMOCNICZE
# ==========================
def ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def is_black_variant(path: str) -> bool:
    """Wzorzec ma być zawsze w wersji 'czarno'."""
    return "czarno" in os.path.basename(path).lower()


def list_images_recursive(folder: str) -> List[str]:
    out = []
    if not folder or not os.path.exists(folder):
        return out
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def load_gray_safe(path: str) -> Optional[np.ndarray]:
    """
    Bezpieczne wczytywanie BMP/JPG/PNG.
    Najpierw imdecode (często rozwiązuje problemy z BMP),
    potem fallback na imread.
    """
    try:
        with open(path, "rb") as stream:
            bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def get_base_id(path_or_name: str) -> str:
    """
    Lekko defensywnie:
    bierze pierwszą część nazwy przed spacją.
    """
    name = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(name)
    base = stem.split()[0]
    return base.lower()


def crop_robust_count(image_path_or_img, from_file=True):
    """
    Twój sprawdzony crop:
    odcina białe tło i zostawia auto.
    """
    if from_file:
        img = load_gray_safe(image_path_or_img)
    else:
        img = image_path_or_img

    if img is None:
        return None

    h, w = img.shape

    floor_cutoff = int(h * 0.85)
    ceiling_cutoff = 5
    analysis_zone = img[ceiling_cutoff:floor_cutoff, :]

    binary_mask = (analysis_zone < 240).astype(np.uint8)

    col_counts = np.sum(binary_mask, axis=0)
    valid_cols = np.where(col_counts > 30)[0]

    if len(valid_cols) == 0:
        return img

    x_min, x_max = valid_cols[0], valid_cols[-1]

    row_counts = np.sum(binary_mask[:, x_min:x_max], axis=1)
    valid_rows = np.where(row_counts > 30)[0]

    if len(valid_rows) == 0:
        y_min = 0
    else:
        y_min = valid_rows[0] + ceiling_cutoff

    y_max = h

    pad = 50
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    y_min = max(0, y_min - pad)

    return img[y_min:y_max, x_min:x_max]


# ==========================
# 1) SZUKANIE NAJLEPSZEGO WZORCA
# ==========================
def find_best_template_crop(
    input_crop: np.ndarray,
    template_paths: List[str],
    search_size: Tuple[int, int] = SEARCH_SIZE,
    exclude_base: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[str], float]:
    """
    Zwraca:
    - best_template_crop
    - best_path
    - best_score (MSE na miniaturach)
    """
    if input_crop is None or not template_paths:
        return None, None, float("inf")

    inp_thumb = cv2.resize(input_crop, search_size).astype("float32")

    best_score = float("inf")
    best_crop = None
    best_path = None

    for p in template_paths:
        if exclude_base and get_base_id(p) == exclude_base:
            continue

        img = load_gray_safe(p)
        if img is None:
            continue

        t_crop = crop_robust_count(img, from_file=False)
        if t_crop is None:
            continue

        t_thumb = cv2.resize(t_crop, search_size).astype("float32")

        # MSE
        score = np.mean((inp_thumb - t_thumb) ** 2)

        if score < best_score:
            best_score = score
            best_crop = t_crop
            best_path = p

    return best_crop, best_path, best_score


# ==========================
# 2) DIFF + MASKA (Twoja logika)
# ==========================
def generate_diff_and_mask(img_input: np.ndarray, img_template: np.ndarray):
    """
    Zwraca:
    - diff (absdiff po blurze)
    - final_mask (po morfologii + filtracji blobów)
    """
    h, w = img_input.shape
    if img_template.shape != (h, w):
        img_template = cv2.resize(img_template, (w, h))

    k = ensure_odd(BLUR_K)
    i_blur = cv2.GaussianBlur(img_input, (k, k), 0)
    t_blur = cv2.GaussianBlur(img_template, (k, k), 0)

    diff = cv2.absdiff(i_blur, t_blur)

    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=ERODE_ITER)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITER)

    # Wycięcie dołu
    cut_bot = int(h * BOTTOM_CUT_FRAC)
    mask[cut_bot:, :] = 0

    # Filtrowanie blobów
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]

        if area < MIN_AREA or area > MAX_AREA:
            continue

        aspect = w_box / float(h_box + 1e-6)
        if not (MIN_AR <= aspect <= MAX_AR):
            continue

        rect_area = w_box * h_box
        fill = area / float(rect_area + 1e-6)
        if fill < MIN_FILL:
            continue

        if (
            x <= BORDER_MARGIN or y <= BORDER_MARGIN
            or x + w_box >= w - BORDER_MARGIN
            or y + h_box >= h - BORDER_MARGIN
        ):
            continue

        final_mask[labels == label] = 255

    return diff, final_mask


# ==========================
# 3) PIPELINE GŁÓWNY
# ==========================
def run_diff_pipeline(
    input_path: str,
    template_folder: str,
    out_dir: str = OUT_DIR_DEFAULT,
    only_black_templates: bool = True,
    use_crop: bool = True,
    exclude_same_base: bool = True
) -> Dict:
    """
    Pipeline:
    dowolne zdjęcie → najlepszy wzorzec (z 'czarno') → diff/maska → zapis .bmp

    Zwraca słownik z informacjami o wynikach.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Nie istnieje plik wejściowy: {input_path}")

    os.makedirs(out_dir, exist_ok=True)

    # 1) Wczytaj input
    img_in = load_gray_safe(input_path)
    if img_in is None:
        raise ValueError(f"Nie mogę wczytać: {input_path}")

    # 2) Crop input
    if use_crop:
        in_crop = crop_robust_count(img_in, from_file=False)
        if in_crop is None:
            raise ValueError(f"Nie udało się wykonać crop dla: {input_path}")
    else:
        in_crop = img_in

    # 3) Zbierz wzorce rekurencyjnie
    templates = list_images_recursive(template_folder)

    if only_black_templates:
        templates = [p for p in templates if is_black_variant(p)]

    if not templates:
        raise ValueError(
            "Brak kandydatów wzorców. "
            "Sprawdź czy w TEMPLATE_FOLDER istnieją pliki z dopiskiem 'czarno'."
        )

    # 4) Znajdź najlepszego bliźniaka
    exclude_base = get_base_id(input_path) if exclude_same_base else None
    best_crop, best_path, best_score = find_best_template_crop(
        in_crop, templates, search_size=SEARCH_SIZE, exclude_base=exclude_base
    )

    if best_crop is None or best_path is None:
        raise ValueError("Nie udało się znaleźć pasującego wzorca.")

    # 5) Diff + maska
    diff, mask = generate_diff_and_mask(in_crop, best_crop)

    # 6) Zapis
    stem = os.path.splitext(os.path.basename(input_path))[0]

    diff_path = os.path.join(out_dir, f"{stem}_diff.bmp")
    mask_path = os.path.join(out_dir, f"{stem}_mask.bmp")

    cv2.imwrite(diff_path, diff)
    cv2.imwrite(mask_path, mask)

    return {
        "input": input_path,
        "template_folder": template_folder,
        "best_template": best_path,
        "best_score_mse": float(best_score),
        "diff_path": diff_path,
        "mask_path": mask_path,
        "out_dir": out_dir
    }

if __name__ == "__main__":
    INPUT_IMAGE = "/Users/igorrudolf/hacknation/NAUKA/brudne/202511190113/48001F003202511190113.bmp"
    TEMPLATE_FOLDER = "/Users/igorrudolf/hacknation/NAUKA/czyste"

    info = run_diff_pipeline(
        input_path=INPUT_IMAGE,
        template_folder=TEMPLATE_FOLDER,
        out_dir="WYNIKI_DIFF_PIPELINE",
        only_black_templates=True, 
        use_crop=True,
        exclude_same_base=True
    )

    print("\nPIPELINE ZAKOŃCZONY")
    for k, v in info.items():
        print(f"{k}: {v}")
