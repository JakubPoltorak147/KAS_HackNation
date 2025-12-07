import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import json
import random

# ================= KONFIGURACJA =================
DIRTY_FOLDER = "/Users/igorrudolf/hacknation/NAUKA/brudne" 
CLEAN_FOLDER = "/Users/igorrudolf/hacknation/NAUKA/czyste"
OUTPUT_JSON_DIR = "annotations_json_final"

# Rozmiar pojedynczego panelu (będą 4, układ 2x2)
PANEL_W = 600
PANEL_H = 350
# ================================================

# ================================================
# LOGIKA PRZETWARZANIA OBRAZU
# ================================================
def get_base_id(path_or_name: str) -> str:
    name = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(name)
    base = stem.split()[0]
    return base.lower()

def crop_robust_count(image_path_or_img, from_file=True):
    if from_file:
        try:
            with open(image_path_or_img, "rb") as stream:
                bytes_data = bytearray(stream.read())
                numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
    else:
        img = image_path_or_img

    if img is None: return None
    
    h, w = img.shape
    
    # 1. Strefa analizy
    floor_cutoff = int(h * 0.85)
    ceiling_cutoff = 5 
    analysis_zone = img[ceiling_cutoff:floor_cutoff, :]
    
    # 2. Progowanie
    binary_mask = (analysis_zone < 240).astype(np.uint8)
    
    # 3. Zliczanie X
    col_counts = np.sum(binary_mask, axis=0)
    valid_cols = np.where(col_counts > 30)[0]
    
    if len(valid_cols) == 0: return img 

    x_min, x_max = valid_cols[0], valid_cols[-1]
    
    # 4. Zliczanie Y
    row_counts = np.sum(binary_mask[:, x_min:x_max], axis=1)
    valid_rows = np.where(row_counts > 30)[0]
    
    if len(valid_rows) == 0:
        y_min = 0
    else:
        y_min = valid_rows[0] + ceiling_cutoff

    y_max = h

    # 5. Marginesy
    pad = 50
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    y_min = max(0, y_min - pad)
    
    return img[y_min:y_max, x_min:x_max]

def generate_full_analysis(img_dirty, img_clean):
    """
    Zwraca dwa obrazy:
    1. Raw Diff (Surowa różnica)
    2. Final Mask (Po czyszczeniu i logice blobów)
    """
    # Dopasowanie wymiarów
    h, w = img_dirty.shape
    if img_clean.shape != (h, w):
        img_clean = cv2.resize(img_clean, (w, h))

    # Rozmycie
    k = 5
    d_blur = cv2.GaussianBlur(img_dirty, (k, k), 0)
    c_blur = cv2.GaussianBlur(img_clean, (k, k), 0)

    # 1. RAW DIFF
    diff = cv2.absdiff(d_blur, c_blur)

    # 2. PROCESOWANIE DO MASKI
    _, mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=4)

    # Usuwanie dołu (pasek skanera)
    cut_bot = int(h * 0.93)
    mask[cut_bot:, :] = 0

    # Filtrowanie Blobów (Twoja logika)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)

    MIN_AREA = 200          
    MAX_AREA = 15000        
    MIN_FILL = 0.15         
    MIN_AR, MAX_AR = 0.4, 4.0   
    BORDER_MARGIN = 5           

    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]

        if area < MIN_AREA or area > MAX_AREA: continue
        aspect = w_box / float(h_box + 1e-6)
        if not (MIN_AR <= aspect <= MAX_AR): continue
        rect_area = w_box * h_box
        fill = area / float(rect_area + 1e-6)
        if fill < MIN_FILL: continue
        if (x <= BORDER_MARGIN or y <= BORDER_MARGIN or x + w_box >= w - BORDER_MARGIN or y + h_box >= h - BORDER_MARGIN): continue

        final_mask[labels == label] = 255

    return diff, final_mask

# ================================================
# APLIKACJA GUI (4 PANELE)
# ================================================
class QuadLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("KAS Labeler 4-Panel (Crop + Mask Logic)")
        # Ustawiamy rozmiar okna dynamicznie
        self.root.geometry(f"{PANEL_W*2 + 50}x{PANEL_H*2 + 100}")

        os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

        # 1. Indeksowanie
        print("🔍 Skanowanie folderów...")
        self.image_list = []
        for root_dir, _, files in os.walk(DIRTY_FOLDER):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')) and "czarno" not in file.lower():
                    self.image_list.append(os.path.join(root_dir, file))
        self.image_list.sort()

        self.clean_files_cache = []
        for root_dir, _, files in os.walk(CLEAN_FOLDER):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                    self.clean_files_cache.append(os.path.join(root_dir, file))
        self.clean_files_cache.sort()

        if not self.image_list:
            messagebox.showerror("Błąd", "Brak zdjęć w folderze brudne!")
            root.destroy()
            return

        # Stan
        self.current_idx = 0
        self.bboxes = [] 
        self.current_crop_dims = (0, 0)
        
        # UI scaling
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Mouse
        self.start_x = None
        self.start_y = None
        self.curr_rect = None

        self.setup_ui()
        self.load_current_image()

    def setup_ui(self):
        # Top Bar
        top = tk.Frame(self.root, bg="#eee", height=50)
        top.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_info = tk.Label(top, text="...", bg="#eee", font=("Arial", 11, "bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        btn_frame = tk.Frame(top, bg="#eee")
        btn_frame.pack(side=tk.RIGHT)
        
        tk.Button(btn_frame, text="< Poprzedni (A)", command=self.prev_img).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Wyczyść (C)", command=self.clear_boxes, bg="#ffcccc").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cofnij (Z)", command=self.undo, bg="#ffffcc").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="ZAPISZ (S)", command=self.save_json_manual, bg="#ccccff", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Następny (D) >", command=self.next_img, bg="#ccffcc").pack(side=tk.LEFT, padx=5)

        # Main Grid (2x2)
        grid_frame = tk.Frame(self.root, bg="#202020")
        grid_frame.pack(fill=tk.BOTH, expand=True)

        # --- ROW 0: DIRTY & CLEAN ---
        
        # 1. Dirty (Top-Left)
        f1 = tk.Frame(grid_frame, bg="#202020", bd=1, relief=tk.SUNKEN)
        f1.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        tk.Label(f1, text="1. BADANY (Crop) - Rysuj tutaj", fg="#ff5555", bg="#202020").pack(side=tk.TOP)
        self.canvas = tk.Canvas(f1, bg="#000", cursor="cross", width=PANEL_W, height=PANEL_H, highlightthickness=0)
        self.canvas.pack()

        # 2. Clean (Top-Right)
        f2 = tk.Frame(grid_frame, bg="#202020", bd=1, relief=tk.SUNKEN)
        f2.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
        tk.Label(f2, text="2. WZORZEC (Dopasowany)", fg="#55ff55", bg="#202020").pack(side=tk.TOP)
        self.lbl_clean = tk.Label(f2, bg="#000", width=PANEL_W, height=PANEL_H)
        self.lbl_clean.pack()

        # --- ROW 1: RAW DIFF & MASK ---

        # 3. Raw Diff (Bottom-Left)
        f3 = tk.Frame(grid_frame, bg="#202020", bd=1, relief=tk.SUNKEN)
        f3.grid(row=1, column=0, padx=2, pady=2, sticky="nsew")
        tk.Label(f3, text="3. SUROWA RÓŻNICA (Ciemne)", fg="#aaaaaa", bg="#202020").pack(side=tk.TOP)
        self.lbl_diff = tk.Label(f3, bg="#000", width=PANEL_W, height=PANEL_H)
        self.lbl_diff.pack()

        # 4. Final Mask (Bottom-Right)
        f4 = tk.Frame(grid_frame, bg="#202020", bd=1, relief=tk.SUNKEN)
        f4.grid(row=1, column=1, padx=2, pady=2, sticky="nsew")
        tk.Label(f4, text="4. MASKA ANOMALII (Filtrowana)", fg="#5555ff", bg="#202020").pack(side=tk.TOP)
        self.lbl_mask = tk.Label(f4, bg="#000", width=PANEL_W, height=PANEL_H)
        self.lbl_mask.pack()

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_rclick)
        
        self.root.bind("a", lambda e: self.prev_img())
        self.root.bind("d", lambda e: self.next_img())
        self.root.bind("s", lambda e: self.save_json_manual())
        self.root.bind("z", lambda e: self.undo())
        self.root.bind("c", lambda e: self.clear_boxes())

    def find_best_match_and_crop(self, dirty_path, img_dirty_crop):
        dirty_base = get_base_id(dirty_path)
        search_size = (64, 64)
        img_dirty_thumb = cv2.resize(img_dirty_crop, search_size).astype("float32")

        best_score = float('inf')
        best_clean_crop = None
        
        candidates = [f for f in self.clean_files_cache if get_base_id(f) != dirty_base]
        if not candidates: candidates = self.clean_files_cache

        # Optymalizacja (30 losowych)
        search_pool = random.sample(candidates, 30) if len(candidates) > 30 else candidates

        for path in search_pool:
            clean_crop = crop_robust_count(path)
            if clean_crop is None: continue
            
            clean_thumb = cv2.resize(clean_crop, search_size).astype("float32")
            score = np.mean((img_dirty_thumb - clean_thumb) ** 2)
            
            if score < best_score:
                best_score = score
                best_clean_crop = clean_crop

        return best_clean_crop

    def cv2_to_pil_padded(self, cv_img):
        """Skaluje do PANEL_W x PANEL_H z paddingiem."""
        h, w = cv_img.shape[:2]
        tw, th = PANEL_W, PANEL_H
        
        ratio = min(tw/w, th/h)
        nw, nh = int(w*ratio), int(h*ratio)
        
        pil = Image.fromarray(cv_img)
        pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
        
        new_img = Image.new("RGB", (tw, th), (32, 32, 32))
        ox = (tw - nw) // 2
        oy = (th - nh) // 2
        
        new_img.paste(pil, (ox, oy))
        return ImageTk.PhotoImage(new_img), ratio, ox, oy

    def load_current_image(self):
        path = self.image_list[self.current_idx]
        fname = os.path.basename(path)
        self.lbl_info.config(text=f"[{self.current_idx+1}/{len(self.image_list)}] {fname}")

        # 1. Dirty Crop
        self.img_dirty_crop = crop_robust_count(path, from_file=True)
        if self.img_dirty_crop is None:
            print("Błąd cropowania!")
            self.next_img()
            return

        self.current_crop_dims = self.img_dirty_crop.shape[:2] # h, w

        # 2. Clean Crop
        self.img_clean_crop = self.find_best_match_and_crop(path, self.img_dirty_crop)
        
        # 3. Generate Logic Maps
        if self.img_clean_crop is not None:
            # Generate Diff and Mask
            diff_cv, mask_cv = generate_full_analysis(self.img_dirty_crop, self.img_clean_crop)
        else:
            diff_cv = np.zeros_like(self.img_dirty_crop)
            mask_cv = np.zeros_like(self.img_dirty_crop)

        # 4. Display Logic (Padding)
        
        # TL: Dirty
        self.tk_dirty, self.scale_factor, self.offset_x, self.offset_y = self.cv2_to_pil_padded(self.img_dirty_crop)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_dirty, anchor=tk.NW)

        # TR: Clean
        if self.img_clean_crop is not None:
            tk_clean, _, _, _ = self.cv2_to_pil_padded(self.img_clean_crop)
            self.lbl_clean.config(image=tk_clean)
            self.lbl_clean.image = tk_clean
        else:
            self.lbl_clean.config(image='', text="BRAK")

        # BL: Raw Diff (Dark)
        tk_diff, _, _, _ = self.cv2_to_pil_padded(diff_cv)
        self.lbl_diff.config(image=tk_diff)
        self.lbl_diff.image = tk_diff

        # BR: Final Mask (Dark + White Blobs)
        tk_mask, _, _, _ = self.cv2_to_pil_padded(mask_cv)
        self.lbl_mask.config(image=tk_mask)
        self.lbl_mask.image = tk_mask

        # Load boxes
        self.load_json_annotations(fname)

    # --- Mouse & JSON Logic (Identyczna jak wcześniej) ---
    def screen_to_img(self, sx, sy):
        x = (sx - self.offset_x) / self.scale_factor
        y = (sy - self.offset_y) / self.scale_factor
        h, w = self.current_crop_dims
        return max(0, min(w, x)), max(0, min(h, y))

    def img_to_screen(self, ix, iy):
        sx = ix * self.scale_factor + self.offset_x
        sy = iy * self.scale_factor + self.offset_y
        return sx, sy

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.curr_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#0f0", width=2)

    def on_drag(self, event):
        self.canvas.coords(self.curr_rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        if abs(event.x - self.start_x) < 5:
            self.canvas.delete(self.curr_rect)
            return
        x1, y1 = self.screen_to_img(self.start_x, self.start_y)
        x2, y2 = self.screen_to_img(event.x, event.y)
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        self.bboxes.append(box)
        self.canvas.delete(self.curr_rect)
        self.redraw_boxes()
        self.save_json_auto()

    def on_rclick(self, event):
        mx, my = self.screen_to_img(event.x, event.y)
        for i in reversed(range(len(self.bboxes))):
            x1, y1, x2, y2 = self.bboxes[i]
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self.bboxes.pop(i)
                self.redraw_boxes()
                self.save_json_auto()
                return

    def redraw_boxes(self):
        self.canvas.delete("box")
        for i, box in enumerate(self.bboxes):
            x1, y1, x2, y2 = box
            sx1, sy1 = self.img_to_screen(x1, y1)
            sx2, sy2 = self.img_to_screen(x2, y2)
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#0f0", width=2, tags="box")
            self.canvas.create_text(sx1, sy1-15, text=str(i+1), fill="#0f0", anchor="sw", tags="box", font=("Arial", 10, "bold"))

    def save_json_manual(self):
        self.save_json_auto()
        print(f"✅ Zapisano manualnie: {os.path.basename(self.image_list[self.current_idx])}")

    def save_json_auto(self):
        path = self.image_list[self.current_idx]
        name = os.path.basename(path)
        json_path = os.path.join(OUTPUT_JSON_DIR, os.path.splitext(name)[0] + ".json")
        h, w = self.current_crop_dims
        data = {"image": name, "width": w, "height": h, "annotations": []}
        for box in self.bboxes:
            data["annotations"].append({"label": "anomaly", "bbox": [int(b) for b in box]})
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_json_annotations(self, filename):
        self.bboxes = []
        json_path = os.path.join(OUTPUT_JSON_DIR, os.path.splitext(filename)[0] + ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for item in data.get("annotations", []):
                        self.bboxes.append(item["bbox"])
            except: pass
        self.redraw_boxes()

    def next_img(self):
        self.save_json_auto()
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_current_image()
        else: messagebox.showinfo("Info", "Koniec.")

    def prev_img(self):
        self.save_json_auto()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

    def undo(self):
        if self.bboxes:
            self.bboxes.pop()
            self.redraw_boxes()
            self.save_json_auto()

    def clear_boxes(self):
        self.bboxes = []
        self.redraw_boxes()
        self.save_json_auto()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuadLabeler(root)
    root.mainloop()