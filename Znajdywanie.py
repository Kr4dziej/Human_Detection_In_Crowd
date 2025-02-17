import json
import numpy as np

# Wczytanie danych z pliku JSON
with open('wyniki_json/Faster_R-CNN_nowe.json', 'r') as f:
    dane = json.load(f)

# Przygotowanie list do przechowywania danych o accuracy i nazwach obrazów
accuracy = []
nazwy_obrazow = []

# Przejście przez wszystkie obrazy w danych
for nazwa_obrazu, obraz in dane.items():
    # Dodanie accuracy i nazwy obrazu do odpowiednich list
    accuracy.append(obraz['accuracy'])
    nazwy_obrazow.append(nazwa_obrazu)

# Posortowanie obrazów według accuracy
posortowane_indeksy = np.argsort(accuracy)

# Wybranie 5 obrazów z najwyższą i najniższą accuracy
najwyzsze_accuracy = [nazwy_obrazow[i] for i in posortowane_indeksy[-5:]]
najnizsze_accuracy = [nazwy_obrazow[i] for i in posortowane_indeksy[:5]]

print("Obrazy z 5 najwyższymi wartościami czułości:", najwyzsze_accuracy)
print("Obrazy z 5 najniższymi wartościami czułości:", najnizsze_accuracy)
