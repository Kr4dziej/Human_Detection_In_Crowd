import ast
import numpy as np
import matplotlib.pyplot as plt
import json

# Wczytanie danych z pliku JSON z wynikami detekcji
with open('wyniki_json/YOLOv8_sylwetki_test.json', 'r') as f:
    detection_results = json.load(f)

# Określenie liczby przedziałów
num_bins = 30  # Zmiana tej wartości zwiększa lub zmniejsza liczbę przedziałów

# Określenie przedziałów dla wartości Y
y_bins = np.linspace(0, 768, num_bins + 1)  # Tworzenie równo rozmieszczonych przedziałów od 0 do wysokości obrazu

# Przygotowanie listy do przechowywania wyników detekcji dla każdego przedziału Y
bin_results = [0] * num_bins
bin_counts = [0] * num_bins

# Przejście przez wszystkie obrazy w danych detekcji
for results in detection_results.values():
    # Przejście przez wszystkie punkty w wynikach detekcji
    for point, detected in results.items():
        # Konwersja punktu z powrotem na krotkę
        point = ast.literal_eval(point)

        # Określenie, do którego przedziału Y należy punkt
        bin_index = np.digitize(point[1], y_bins) - 1

        # Dodanie wyniku detekcji do odpowiedniego przedziału
        bin_results[bin_index] += detected
        bin_counts[bin_index] += 1

# Obliczenie skuteczności detekcji dla każdego przedziału Y
bin_efficiency = [result / count if count > 0 else 0 for result, count in zip(bin_results, bin_counts)]

# Wydrukowanie liczby wykrytych i niewykrytych sylwetek w każdym przedziale
for i in range(num_bins):
    print(f'{y_bins[i]}-{y_bins[i+1]}: {bin_results[i]} wykryto {bin_counts[i]-bin_results[i]} nie wykryto')

# Stworzenie wykresu skuteczności detekcji od wartości Y
plt.figure(figsize=(10, 5))
plt.plot(y_bins[:-1], bin_efficiency)
plt.title('Wykres czułości względem położenia sylwetki na składowej Y')
plt.xlabel('Wartość Y')
plt.ylabel('Czułość')
plt.grid(True)
#plt.savefig('wykresy/wykres_skutecznosci_Y.png')
plt.show()
