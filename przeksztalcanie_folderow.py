import os
import shutil

source_folder = r"C:\Users\macie\OneDrive - Politechnika Warszawska\Pulpit\Hackathony\hacknation\NAUKA\brudne"
target_folder = r"C:\Users\macie\OneDrive - Politechnika Warszawska\Pulpit\Hackathony\hacknation\NAUKAbrudne_przeksztalcone"

os.makedirs(target_folder, exist_ok=True)

counter = 1

for root, dirs, files in os.walk(source_folder):
    for file in files:
        source_path = os.path.join(root, file)
        name, ext = os.path.splitext(file)

        # Jeśli plik o tej nazwie już istnieje – zmień nazwę
        target_path = os.path.join(target_folder, file)
        while os.path.exists(target_path):
            target_path = os.path.join(target_folder, f"{name}_{counter}{ext}")
            counter += 1

        shutil.move(source_path, target_path)  # zmień na copy() jeśli chcesz kopiować
        print(f"Przeniesiono: {source_path} → {target_path}")

print("Gotowe!")
