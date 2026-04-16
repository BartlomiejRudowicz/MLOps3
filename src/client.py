import requests
import numpy as np

def main():
    # Adres naszego endpointu, który stworzyliśmy w BentoML
    url = "http://localhost:3000/predict"

    # Tworzymy losowe dane udające wyniki badań jednego pacjenta (1 próbka, 30 cech)
    # W prawdziwym świecie byłyby to ustandaryzowane dane ze szpitala
    sample_data = np.random.rand(1, 30)

    # Przygotowujemy "paczkę" (payload) do wysłania.
    # Konwertujemy tablicę Numpy na standardową listę Pythona, którą można wysłać jako JSON
    payload = {
        "input_data": sample_data.tolist()
    }

    print(f"Wysyłam dane pacjenta do {url}...")
    
    # Wysyłamy zapytanie POST (czyli przesyłamy dane do serwera)
    response = requests.post(url, json=payload)

    # Sprawdzamy odpowiedź
    if response.status_code == 200:
        print("Sukces! Serwer odpowiedział:")
        print(f"Wynik predykcji (0 = łagodny, 1 = złośliwy): {response.json()}")
    else:
        print(f"Błąd! Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()