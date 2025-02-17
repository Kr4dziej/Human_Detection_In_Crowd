import ast
import numpy as np
import matplotlib.pyplot as plt
import json

# Lista plików JSON do wczytania
pliki_json = ['wyniki_json/HOG_sylwetki.json', 'wyniki_json/YOLOv3_sylwetki.json', 'wyniki_json/YOLOv8_sylwetki.json', 'wyniki_json/Faster_R-CNN_nowe_sylwetki.json']

# Legenda
legenda = ["HOG z SVM", "YOLOv3", "YOLOv8", "Faster R-CNN"]

# Określ liczbę przedziałów
num_bins = 30  # Zmień tę wartość, aby zwiększyć lub zmniejszyć liczbę przedziałów

# Określ przedziały dla wartości Y
y_bins = np.linspace(0, 768, num_bins + 1)  # Tworzy równo rozmieszczone przedziały od 0 do wysokości obrazu

plt.figure(figsize=(10, 5))

# Przejdź przez wszystkie pliki JSON
for plik, leg in zip(pliki_json, legenda):
    # Wczytaj dane z pliku JSON
    with open(plik, 'r') as f:
        detection_results = json.load(f)

    # Przygotuj listę do przechowywania wyników detekcji dla każdego przedziału Y
    bin_results = [0] * num_bins
    bin_counts = [0] * num_bins

    # Przejdź przez wszystkie obrazy w danych detekcji
    for results in detection_results.values():
        # Przejdź przez wszystkie punkty w wynikach detekcji
        for point, detected in results.items():
            # Konwertuj punkt z powrotem na krotkę
            point = ast.literal_eval(point)

            # Określ, do którego przedziału Y należy punkt
            bin_index = np.digitize(point[1], y_bins) - 1

            # Dodaj wynik detekcji do odpowiedniego przedziału
            bin_results[bin_index] += detected
            bin_counts[bin_index] += 1

    # Oblicz skuteczność detekcji dla każdego przedziału Y
    bin_efficiency = [result / count if count > 0 else 0 for result, count in zip(bin_results, bin_counts)]

    # Dodaj linię trendu do wykresu
    plt.plot(y_bins[:-1], bin_efficiency, label=leg)

plt.title('Wykres czułości względem położenia sylwetki na składowej Y')
plt.xlabel('Wartość Y')
plt.ylabel('Czułość')
plt.legend()
plt.grid(True)
plt.savefig('wykresy/Polaczone/wykres_skutecznosci_Y.png')
plt.show()
