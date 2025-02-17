import json
import matplotlib.pyplot as plt
import numpy as np

# Lista plików JSON do wczytania
pliki_json = ['wyniki_json/HOG.json', 'wyniki_json/YOLOv3.json', 'wyniki_json/YOLOv8.json', 'wyniki_json/Faster_R-CNN_nowe.json']

# Stopnie wielomianów dla linii trendu
stopnie = [4, 10, 8, 7]

# Legenda
legenda = ["HOG z SVM", "YOLOv3", "YOLOv8", "Faster R-CNN"]

plt.figure(figsize=(10, 5))

# Przejście przez wszystkie pliki JSON
for plik, stopien, leg in zip(pliki_json, stopnie, legenda):
    # Wczytanie danych z pliku JSON
    with open(plik, 'r') as f:
        dane = json.load(f)

    # Przygotowanie listy do przechowywania danych o accuracy, czasie detekcji i liczbie ludzi
    accuracy = []
    czas_detekcji = []
    ludzie = []

    # Przejście przez wszystkie obrazy w danych
    for obraz in dane.values():
        # Dodanie accuracy, czas detekcji i liczbę ludzi do odpowiednich list
        accuracy.append(obraz['accuracy'])
        czas_detekcji.append(obraz['detection_time'])
        ludzie.append(obraz['human_number'])

    # Dodanie linii trendu
    z = np.polyfit(ludzie[1:], accuracy[1:], stopien)
    p = np.poly1d(z)
    ludzie_sort = np.sort(ludzie[1:])
    plt.plot(ludzie_sort, p(ludzie_sort), label=leg)

plt.title('Wykres czułości względem liczby sylwetek na obrazie')
plt.xlabel('Liczba sylwetek na obrazie')
plt.ylabel('Czułość')
plt.legend()
plt.grid(True)
plt.savefig('wykresy/Polaczone/wykres_accuracy.jpg')
plt.show()

# Wykres czasu detekcji
plt.figure(figsize=(10, 5))

# Przejście przez wszystkie pliki JSON
for plik, leg in zip(pliki_json, legenda):
    # Wczytaj dane z pliku JSON
    with open(plik, 'r') as f:
        dane = json.load(f)

    # Przygotowanie listy do przechowywania danych o accuracy, czasie detekcji i liczbie ludzi
    accuracy = []
    czas_detekcji = []
    ludzie = []

    # Przejście przez wszystkie obrazy w danych
    for obraz in dane.values():
        # Dodanie accuracy, czas detekcji i liczbę ludzi do odpowiednich list
        accuracy.append(obraz['accuracy'])
        czas_detekcji.append(obraz['detection_time'])
        ludzie.append(obraz['human_number'])

    # Dodanie linii trendu
    z = np.polyfit(ludzie[1:], czas_detekcji[1:], 3)
    p = np.poly1d(z)
    ludzie_sort = np.sort(ludzie[1:])
    plt.plot(ludzie_sort, p(ludzie_sort), label=leg)

plt.title('Wykres czasu detekcji względem liczby sylwetek na obrazie')
plt.xlabel('Liczba sylwetek na obrazie')
plt.ylabel('Czas detekcji [s]')
plt.legend(bbox_to_anchor=(0.9, 0.9), loc='upper center')
plt.grid(True)
plt.savefig('wykresy/Polaczone/wykres_czas_detekcji.jpg')
plt.show()
