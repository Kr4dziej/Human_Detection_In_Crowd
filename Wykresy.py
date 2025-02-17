import json
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych z pliku JSON
with open('wyniki_json/YOLOv8_test_zmiany_550.json', 'r') as f:
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

# Stworzenie wykresu accuracy od liczby ludzi
plt.figure(figsize=(10, 5))
plt.scatter(ludzie, accuracy, s=10)
#plt.scatter(ludzie[1:], accuracy[1:], s=10)  # Pomiń pierwszy element
# Dodanie linii trendu
z = np.polyfit(ludzie[1:], accuracy[1:], 7)
p = np.poly1d(z)
ludzie_sort = np.sort(ludzie[1:])
plt.plot(ludzie_sort, p(ludzie_sort), "r")
plt.title('Wykres czułości względem liczby sylwetek na obrazie')
plt.xlabel('Liczba sylwetek na obrazie')
plt.ylabel('Czułość')
plt.grid(True)
#plt.savefig('wykresy/Faster R-CNN/wykres_accuracy.jpg')
plt.show()

# Stworzenie wykresu czasu detekcji od liczby ludzi
plt.figure(figsize=(10, 5))
plt.scatter(ludzie, czas_detekcji, s=10)
#plt.scatter(ludzie[1:], czas_detekcji[1:], s=10)  # Pomiń pierwszy element
# Dodanie linii trendu
z = np.polyfit(ludzie[1:], czas_detekcji[1:], 3)
p = np.poly1d(z)
ludzie_sort = np.sort(ludzie[1:])
plt.plot(ludzie_sort, p(ludzie_sort), "r")
plt.title('Wykres czasu detekcji względem liczby sylwetek na obrazie')
plt.xlabel('Liczba sylwetek na obrazie')
plt.ylabel('Czas detekcji [s]')
plt.grid(True)
#plt.savefig('wykresy/Faster R-CNN/wykres_czas_detekcji.jpg')
plt.show()
