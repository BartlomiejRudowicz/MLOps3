# Zadanie 3: Konteneryzacja i Wdrożenie modelu w chmurze (AWS)

## Opis projektu
Celem zadania było wdrożenie produkcyjnego serwisu do predykcji nowotworu piersi (Breast Cancer) przy użyciu frameworka **BentoML**, narzędzia **Docker** oraz infrastruktury chmurowej **AWS EC2**. Projekt obejmuje pełny cykl życia modelu: od pakowania (Bento), przez konteneryzację, aż po udostępnienie publicznego API.

## Architektura i Narzędzia
* **Model:** Sieć neuronowa MLP (Multi-Layer Perceptron) wytrenowana w PyTorch.
* **Serwis:** BentoML (zapewnia obsługę zapytań HTTP i zarządzanie modelem).
* **Konteneryzacja:** Docker (izolacja środowiska).
* **Chmura:** AWS EC2 (instancja `m7i-flex.large`) – wybrana ze względu na wysoką wydajność procesora i odpowiednią ilość pamięci RAM (8 GB) do budowy obrazów.
* **System operacyjny:** Ubuntu 24.04 LTS.

## Struktura plików
```text
MLOps-main/
├── src/
│   ├── model.py          # Definicja architektury MLPClassifier
│   ├── service.py        # Serwis BentoML z poprawkami ścieżek i zabezpieczeń
│   ├── client_aws.py     # Skrypt testowy (klient HTTP)
│   └── __init__.py       # Inicjalizacja pakietu
├── bentofile.yaml        # Konfiguracja budowania Bento (pakiety, modele)
├── requirements.txt      # Zależności Python (zoptymalizowane pod CPU)
└── my_bento.bento        # Wyeksportowany pakiet gotowy do transferu
```

## Instrukcja wdrożenia

### 1. Przygotowanie lokalne (Windows)
Zbudowanie pakietu Bento z uwzględnieniem modelu i odpowiednich ścieżek:
```powershell
$env:PYTHONPATH = ".;./src"
bentoml build
bentoml export breast_cancer_service:latest ./my_bento.bento
```

### 2. Transfer i Konfiguracja AWS
Przesłanie pakietu na serwer i konfiguracja maszyny:
* **Dysk:** Zwiększony do 20 GB (moduły PyTorch i warstwy Dockera wymagają dodatkowej przestrzeni).
* **Sieć:** Otwarty port TCP `3000` w Inbound Rules (Security Groups).
* **Transfer:**
    ```bash
    scp -i "klucz.pem" my_bento.bento ubuntu@IP_SERWERA:~/
    ```

### 3. Konteneryzacja na serwerze
Uruchomienie procesu budowania obrazu OCI na maszynie docelowej:
```bash
bentoml import ~/my_bento.bento
bentoml containerize breast_cancer_service:latest
```

### 4. Uruchomienie produkcyjne
Uruchomienie kontenera na porcie 3000:
```bash
docker run -it --rm -p 3000:3000 breast_cancer_service:[TAG]
```

## Testowanie API
Serwis jest dostępny pod adresem: `http://IP_AWS:3000`. 
Predykcji można dokonać wysyłając żądanie POST na endpoint `/predict` z listą 30 cech numerycznych.
