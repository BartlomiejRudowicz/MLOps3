import requests
import numpy as np

# IP Twojej maszyny m7i-flex
URL = "http://13.60.232.183:3000/predict" 

# Przykładowe dane wejściowe (30 cech - np. średnie wartości dla nowotworu)
# W prawdziwym scenariuszu tu byłyby realne pomiary
sample_data = [[
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]]

payload = {"input_data": sample_data}

print(f"Łączę się z serwerem AWS: {URL}...")

try:
    response = requests.post(URL, json=payload)
    if response.status_code == 200:
        prediction = response.json()
        print("Sukces! Wynik predykcji z chmury:")
        print(f"Klasa: {prediction} (gdzie [1] to złośliwy, [0] to łagodny)")
    else:
        print(f"Błąd serwera! Kod: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Nie udało się połączyć: {e}")