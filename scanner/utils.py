import cv2
import numpy as np
import os
from django.conf import settings
from django.core.files.base import ContentFile
import io
from ultralytics import YOLO

# USTAWIENIE ROZMIARU
IMG_SIZE = 1024 

MODEL_PATH = os.path.join(settings.BASE_DIR, 'scanner', 'ml_models', 'best.pt')

try:
    ai_model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"UWAGA: Nie znaleziono modelu pod ścieżką {MODEL_PATH}. Błąd: {e}")
    ai_model = None

def safe_imread(path):
    """
    Bezpieczne wczytywanie obrazów na Windows.
    Zwraca obraz w BGR (3 kanały).
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        # Wczytujemy jako KOLOR, żeby mieć pewność struktury (H, W, 3)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR) 
        return img
    except Exception as e:
        print(f"Błąd wczytywania pliku {path}: {e}")
        return None

def crop_robust_count(image_path):
    """
    Wycina auto ze zdjęcia.
    Zwraca WYCINEK W SKALI SZAROŚCI, ale w formacie BGR (3 kanały).
    Dzięki temu usuwamy ewentualne kolory/szumy barwne, ale możemy rysować czerwone ramki.
    """
    img_color = safe_imread(image_path)
    if img_color is None:
        print(f"crop_robust_count: Nie udało się wczytać obrazu: {image_path}")
        return None
    
    # 1. Konwersja do skali szarości (tu pozbywamy się koloru)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # --- LOGIKA WYCINANIA (na podstawie jasności pikseli) ---
    h, w = img_gray.shape
    floor_cutoff = int(h * 0.85)
    ceiling_cutoff = 5 
    
    analysis_zone = img_gray[ceiling_cutoff:floor_cutoff, :]
    binary_mask = (analysis_zone < 240).astype(np.uint8)
    
    col_counts = np.sum(binary_mask, axis=0)
    valid_cols = np.where(col_counts > 30)[0]
    
    if len(valid_cols) == 0: return None 

    x_min = valid_cols[0]
    x_max = valid_cols[-1]
    
    row_counts = np.sum(binary_mask[:, x_min:x_max], axis=1)
    valid_rows = np.where(row_counts > 30)[0]
    
    if len(valid_rows) == 0:
        y_min = 0
    else:
        y_min = valid_rows[0] + ceiling_cutoff

    y_max = h 
    pad = 30
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    y_min = max(0, y_min - pad)

    # --- KLUCZOWA ZMIANA ---
    # Mamy img_gray (czarno-białe).
    # Konwertujemy je z powrotem na BGR (GRAY -> BGR).
    # Wizualnie to nadal odcienie szarości, ale technicznie ma 3 kanały.
    # Dzięki temu usunęliśmy wszelkie odcienie kolorów z wejścia, a możemy rysować na czerwono.
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Zwracamy wycinek z tego "szarego" obrazu 3-kanałowego
    return img_gray_3ch[y_min:y_max, x_min:x_max]

def process_scan_request(scan_request_instance, candidates_queryset=None):
    """
    Logika oparta o sieć neuronową (.pt).
    Analizujemy CROP (wycinek) i na nim rysujemy.
    """
    # 1. Ścieżka do pliku
    image_path = scan_request_instance.image.path
    
    if ai_model is None:
        print("Błąd: Model AI nie został załadowany.")
        return 0, None

    # 2. PRZYGOTOWANIE OBRAZU (CROP)
    # Teraz ta funkcja zwraca obraz w skali szarości (jako 3 kanały BGR)
    img_cropped = crop_robust_count(image_path)
    
    if img_cropped is None:
        print("Błąd: Nie udało się wyciąć auta (lub wczytać pliku).")
        return 0, None

# 3. INFERENCJA
    results = ai_model.predict(
        source=img_cropped, 
        conf=0.35, 
        imgsz=IMG_SIZE, 
        verbose=False
    )
    
    result = results[0]
    anomalies_count = 0
    detections_data = [] # Lista na dane dla HTML
    
    # Pobieramy wymiary obrazka, żeby liczyć procenty
    img_h, img_w = img_cropped.shape[:2]

    if len(result.boxes) > 0:
        for box in result.boxes:
            # Koordynaty pikselowe
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # --- RYSOWANIE (Tylko ramka, BEZ NAPISÓW) ---
            # Rysujemy samą ramkę na czerwono
            cv2.rectangle(img_cropped, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # --- PRZYGOTOWANIE DANYCH DLA HTML (Procenty) ---
            # Dzięki procentom ramki będą pasować na każdym ekranie (responsywność)
            box_data = {
                'x': (x1 / img_w) * 100,      # Left %
                'y': (y1 / img_h) * 100,      # Top %
                'w': ((x2 - x1) / img_w) * 100, # Width %
                'h': ((y2 - y1) / img_h) * 100, # Height %
                'conf': int(conf * 100)       # Pewność %
            }
            detections_data.append(box_data)
            
            anomalies_count += 1
    else:
        print("Model nie wykrył żadnych anomalii.")

    scan_request_instance.matched_clean_image = None
    
    # 5. ZAPIS WYNIKU
    is_success, buffer = cv2.imencode(".jpg", img_cropped)
    io_buf = io.BytesIO(buffer)
    
    file_name = f"result_ai_{scan_request_instance.id}.jpg"
    
    # Zwracamy teraz TRZY rzeczy: licznik, plik, i dane JSON
    return anomalies_count, ContentFile(io_buf.getvalue(), name=file_name), detections_data


# scanner/utils.py (dodaj na końcu lub w odpowiednim miejscu)

# --- KONFIGURACJA Z DIFF_PIPELINE ---
SEARCH_SIZE = (128, 128)
BLUR_K = 5
DIFF_THRESH = 45

def generate_morphology_diff(scan_instance, candidates_queryset):
    """
    Implementacja logiki z diff_pipeline.py.
    1. Znajduje najlepszy wzorzec (MSE).
    2. Generuje obraz różnicowy (Diff).
    3. Zwraca obraz Diff jako ContentFile.
    """
    
    # 1. Wczytaj i wytnij obraz wejściowy
    img_input_color = safe_imread(scan_instance.image.path)
    if img_input_color is None:
        return None

    # Konwersja na szary do obliczeń (algorytm działa na grayscale)
    img_input_gray = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)
    
    # Używamy Twojej funkcji cropującej (zakładam, że zwraca ona cropa)
    # Musimy wyciąć wersję szarą
    input_crop = crop_robust_count(scan_instance.image.path) 
    # UWAGA: crop_robust_count w poprzedniej wersji zwracał BGR. 
    # Jeśli zwraca BGR, konwertujemy na GRAY:
    if len(input_crop.shape) == 3:
        input_crop = cv2.cvtColor(input_crop, cv2.COLOR_BGR2GRAY)

    # Miniaturka wejścia do szukania
    inp_thumb = cv2.resize(input_crop, SEARCH_SIZE).astype("float32")

    best_score = float("inf")
    best_template_crop = None
    
    # 2. Szukanie najlepszego wzorca w bazie
    for candidate in candidates_queryset:
        try:
            # Wczytaj wzorzec
            cand_path = candidate.image.path
            cand_img = safe_imread(cand_path)
            if cand_img is None: continue

            # Crop wzorca
            cand_crop_color = crop_robust_count(cand_path)
            if cand_crop_color is None: continue
            
            # Konwersja na szary
            if len(cand_crop_color.shape) == 3:
                cand_crop = cv2.cvtColor(cand_crop_color, cv2.COLOR_BGR2GRAY)
            else:
                cand_crop = cand_crop_color

            # Resize do miniatury
            cand_thumb = cv2.resize(cand_crop, SEARCH_SIZE).astype("float32")

            # MSE
            score = np.mean((inp_thumb - cand_thumb) ** 2)

            if score < best_score:
                best_score = score
                best_template_crop = cand_crop

        except Exception as e:
            print(f"Błąd przy wzorcu {candidate.id}: {e}")
            continue

    if best_template_crop is None:
        print("Nie znaleziono pasującego wzorca do morfologii.")
        return None

    # 3. Generowanie DIFF (Przedostatni krok algorytmu)
    # Dopasowanie rozmiarów (na wypadek drobnych różnic w cropie)
    h, w = input_crop.shape
    if best_template_crop.shape != (h, w):
        best_template_crop = cv2.resize(best_template_crop, (w, h))

    # Blur (usuwanie szumu matrycy)
    k = BLUR_K if BLUR_K % 2 == 1 else BLUR_K + 1
    i_blur = cv2.GaussianBlur(input_crop, (k, k), 0)
    t_blur = cv2.GaussianBlur(best_template_crop, (k, k), 0)

    # Różnica absolutna (To jest ten obraz, o który prosisz!)
    diff_img = cv2.absdiff(i_blur, t_blur)

    # Opcjonalnie: Możemy go trochę wzmocnić wizualnie (np. mnożąc x2), 
    # żeby różnice były lepiej widoczne dla ludzkiego oka
    diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

    # Enkodowanie do pliku
    is_success, buffer = cv2.imencode(".jpg", diff_img)
    io_buf = io.BytesIO(buffer)
    
    return ContentFile(io_buf.getvalue(), name=f"diff_{scan_instance.id}.jpg")