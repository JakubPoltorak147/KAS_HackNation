# KAS HackNation - System Detekcji Anomalii w Obrazach Rentgenowskich

System webowy oparty na Django do automatycznej detekcji anomalii w obrazach rentgenowskich pojazdów, stworzony podczas hackathonu HackNation.

## 📋 Opis projektu

Aplikacja wykorzystuje techniki machine learning do analizy obrazów rentgenowskich pojazdów w celu wykrywania potencjalnych zagrożeń i anomalii. System przetwarza zarówno "czyste" obrazy referencyjne, jak i obrazy zawierające anomalie, aby nauczyć model rozróżniania między nimi.

## 🚀 Funkcjonalności

- **Automatyczna detekcja anomalii** - wykrywanie nieprawidłowości w skanach rentgenowskich
- **Pipeline przetwarzania obrazów** - transformacja i przygotowanie danych wejściowych
- **System adnotacji** - zarządzanie adnotacjami w formacie JSON dla obrazów treningowych
- **Interfejs webowy** - aplikacja Django umożliwiająca łatwe korzystanie z systemu
- **Analiza różnicowa** - porównywanie obrazów "czystych" i "brudnych"

## 🏗️ Struktura projektu

```
KAS_HackNation/
│
├── scanner/                          # Główna aplikacja Django
├── core/                             # Konfiguracja projektu Django
│
├── brudne_przeksztalcone/           # Obrazy z anomaliami (przetworzone)
├── czyste_przeksztalcone/           # Obrazy referencyjne (przetworzone)
│
├── json_annotations_clean_final/    # Adnotacje dla czystych obrazów
├── json_annotations_dirty_final/    # Adnotacje dla obrazów z anomaliami
│
├── anomaly_app_script.py            # Skrypt aplikacji do detekcji anomalii
├── diff_pipeline.py                 # Pipeline analizy różnicowej
├── przeksztalcanie_folderow.py      # Skrypt transformacji katalogów
│
├── detecting_anomalies.ipynb        # Notebook badawczy - detekcja anomalii
├── podejscie_temporary.ipynb        # Notebook z eksperymentami
├── tmp.ipynb                        # Notebook roboczy
│
├── manage.py                        # Skrypt zarządzania Django
└── requirements.txt                 # Zależności projektu
```

## 🔧 Technologie

- **Backend**: Django
- **Machine Learning**: PyTorch / TensorFlow (do detekcji anomalii)
- **Przetwarzanie obrazów**: OpenCV, PIL/Pillow
- **Analiza danych**: NumPy, Pandas
- **Notebooki**: Jupyter

## 📦 Instalacja

### Wymagania wstępne

- Python 3.8+
- pip
- virtualenv (zalecane)

### Kroki instalacji

1. **Sklonuj repozytorium**
```bash
git clone https://github.com/JakubPoltorak147/KAS_HackNation.git
cd KAS_HackNation
```

2. **Utwórz i aktywuj środowisko wirtualne**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate  # Windows
```

3. **Zainstaluj zależności**
```bash
pip install -r requirements.txt
```

4. **Wykonaj migracje Django**
```bash
python manage.py migrate
```

5. **Uruchom serwer deweloperski**
```bash
python manage.py runserver
```

Aplikacja będzie dostępna pod adresem: `http://127.0.0.1:8000`

## 💻 Użytkowanie

### Interfejs webowy

1. Uruchom serwer Django zgodnie z instrukcją instalacji
2. Otwórz przeglądarkę i przejdź do `http://127.0.0.1:8000`
3. Wgraj obraz rentgenowski do analizy
4. System automatycznie wykryje anomalie i wyświetli wyniki

### Skrypty pomocnicze

**Detekcja anomalii:**
```bash
python anomaly_app_script.py
```

**Pipeline różnicowy:**
```bash
python diff_pipeline.py
```

**Transformacja katalogów:**
```bash
python przeksztalcanie_folderow.py
```

### Jupyter Notebooks

Projekt zawiera notebooki badawcze do eksperymentowania z różnymi podejściami:

```bash
jupyter notebook detecting_anomalies.ipynb
```

## 🧠 Model detekcji anomalii

System wykorzystuje podejście oparte na analizie różnicowej między obrazami referencyjnymi (czystymi) a obrazami testowymi. Model jest trenowany na zbiorze danych zawierającym:

- **Czyste obrazy** - normalne skany rentgenowskie bez anomalii
- **Brudne obrazy** - skany zawierające kontrabandę lub inne nieprawidłowości

Adnotacje w formacie JSON zawierają informacje o lokalizacji i typie wykrytych anomalii.

## 📊 Format danych

### Struktura adnotacji JSON

```json
{
  "image_id": "example_001",
  "annotations": [
    {
      "type": "anomaly",
      "bbox": [x, y, width, height],
      "confidence": 0.95
    }
  ]
}
```

## 🛠️ Rozwój projektu

### Dodawanie nowych funkcjonalności

1. Stwórz nową gałąź
```bash
git checkout -b feature/nazwa-funkcjonalnosci
```

2. Wprowadź zmiany i przetestuj
3. Stwórz Pull Request

---

**Uwaga**: Projekt był tworzony w ramach hackathonu i może wymagać dodatkowej konfiguracji dla środowiska produkcyjnego.
