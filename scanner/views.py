from django.shortcuts import render, redirect, get_object_or_404
from .models import ScanRequest, CleanImage
from .forms import ScanRequestForm
from .utils import process_scan_request, crop_robust_count
from django.core.files.base import ContentFile
from .utils import process_scan_request, crop_robust_count, generate_morphology_diff # <--- Import
from .models import CleanImage # <--- Import jeśli nie ma

def upload_scan(request):
    """Strona główna: Upload pliku + Uruchomienie analizy"""
    if request.method == 'POST':
        form = ScanRequestForm(request.POST, request.FILES)
        if form.is_valid():
            # 1. Zapisz wstępnie obiekt
            scan_obj = form.save(commit=False)
            scan_obj.status = 'processing'
            scan_obj.save()

            try:
                input_path = scan_obj.image.path
                cropped_input = crop_robust_count(input_path)
                
                if cropped_input is not None:
                    # --- POPRAWKA BŁĘDU ---
                    # Obraz kolorowy ma 3 wymiary (H, W, Channels). Pobieramy tylko dwa pierwsze.
                    h, w = cropped_input.shape[:2] 
                    
                    target_ar = w / h
                    
                    # --- DEBUGOWANIE START ---
                    print(f"\n=== DEBUGOWANIE SKANU ===")
                    print(f"1. Plik wejściowy: {scan_obj.image.name}")
                    print(f"2. Wykryte wymiary (Crop): {w}x{h}")
                    
                    # Logika szukania kandydatów w bazie jest teraz opcjonalna (AI tego nie potrzebuje),
                    # ale zostawiamy ją, żeby nie psuć reszty kodu widoku.
                    
                    candidates = CleanImage.objects.all()
                    # Wywołanie AI - przekazujemy None jako kandydatów
                    anomalies_count, result_file, detections_data = process_scan_request(scan_obj, None)

                    diff_file = generate_morphology_diff(scan_obj, candidates)
                    
                    if result_file:
                        filename = f"result_{scan_obj.id}.jpg"
                        scan_obj.result_image.save(filename, result_file, save=True)
                        
                        if diff_file:
                            scan_obj.difference_image.save(f"diff_{scan_obj.id}.jpg", diff_file, save=False)
                        
                        scan_obj.anomalies_found = anomalies_count
                        scan_obj.detection_data = detections_data  # <--- ZAPISUJEMY DANE JSON
                        scan_obj.status = 'completed'
                        scan_obj.save(update_fields=['anomalies_found', 'status', 'detection_data'])
                        
                        print(f"SUKCES: Zapisano wynik w: {scan_obj.result_image.path}")
                    else:
                        print("BŁĄD: process_scan_request nie zwrócił pliku.")
                        scan_obj.status = 'error'
                        scan_obj.save()

                else:
                    print("BŁĄD: crop_robust_count zwrócił None (nie wykryto auta na wejściu).")
                    scan_obj.status = 'error'
                
            
            except Exception as e:
                # Wypisujemy pełny traceback błędu w konsoli, żeby łatwiej debugować
                import traceback
                traceback.print_exc()
                print(f"Critical Error w views.py: {e}")
                scan_obj.status = 'error'
            
            scan_obj.save()
            return redirect('scan_result', pk=scan_obj.pk)
    else:
        form = ScanRequestForm()
    
    return render(request, 'scanner/upload.html', {'form': form})

def scan_result(request, pk):
    """Strona wyniku"""
    scan = get_object_or_404(ScanRequest, pk=pk)
    
    if request.method == 'POST':
        label = request.POST.get('confirmed_label')
        comment = request.POST.get('user_comments')
        
        if label:
            scan.confirmed_label = label
            scan.user_comments = comment
            scan.save()
            
            # --- LOGIKA DATA FLYWHEEL (Douczanie) ---
            if label == 'clean':
                original_filename = scan.image.name.split('/')[-1]
                exists = CleanImage.objects.filter(image__icontains=original_filename).exists()
                
                if not exists:
                    try:
                        new_clean = CleanImage()
                        new_clean.title = f"Auto-Learned: {original_filename}"
                        with scan.image.open('rb') as f:
                            new_clean.image.save(f"learned_{original_filename}", ContentFile(f.read()))
                        new_clean.save()
                        print(f"SYSTEM DOUCZONY: Dodano nowe czyste auto do bazy: {new_clean.title}")
                    except Exception as e:
                        print(f"Błąd podczas douczania (clean): {e}")

            return render(request, 'scanner/result.html', {'scan': scan, 'saved': True})

    return render(request, 'scanner/result.html', {'scan': scan})