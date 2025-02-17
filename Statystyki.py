import json

# Wczytanie danych z pliku JSON
with open('wyniki_json/YOLOv8_test_zmiany_550.json', 'r') as f:
    dane = json.load(f)

# Inicjalizacja zmiennej do przechowywania sumy accuracy
suma_accuracy = 0
suma_czas = 0

# Przejście przez wszystkie obrazy w danych
for obraz in dane.values():
    # Dodanie accuracy obrazu do sumy
    suma_accuracy += obraz['accuracy']
    suma_czas += obraz['detection_time']

# Obliczenie średniej accuracy
srednia_accuracy = suma_accuracy / len(dane)
sredni_czas = suma_czas / len(dane)

print(f'Średnia czułość: {srednia_accuracy}')
print(f'Średni czas: {sredni_czas}')
