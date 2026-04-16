import os
import sys
import torch
import bentoml
import functools
import numpy as np

# --- DYNAMICZNA NAPRAWA ŚCIEŻEK (DLA DOCKERA) ---
# Pobieramy ścieżkę do folderu, w którym jest ten plik (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Pobieramy ścieżkę do folderu nadrzędnego (główny folder Bento)
parent_dir = os.path.dirname(current_dir)

# Dodajemy oba foldery do ścieżek wyszukiwania na samym początku
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- FIX DLA PYTORCH 2.6+ ---
torch.load = functools.partial(torch.load, weights_only=False)

# --- INTELIGENTNY IMPORT MODELU ---
try:
    # Próba 1: Bezpośrednio (jeśli jesteśmy w src)
    from model import MLPClassifier
except ImportError:
    try:
        # Próba 2: Przez pakiet src
        from src.model import MLPClassifier
    except ImportError:
        # Próba 3: Ostateczna - jeśli wszystko inne zawiedzie
        import model
        MLPClassifier = model.MLPClassifier

@bentoml.service(
    name="breast_cancer_service",
    traffic={"timeout": 60},
)
class BreastCancerService:
    # Model pobieramy wewnątrz init, by uniknąć błędów przy budowaniu
    def __init__(self):
        self.bento_model = bentoml.models.get("breast_cancer_mlp:latest")
        self.model = bentoml.pytorch.load_model(self.bento_model)
        self.model.eval()

    @bentoml.api
    def predict(self, input_data: list) -> list:
        inputs = torch.tensor(input_data).float()
        with torch.no_grad():
            outputs = self.model(inputs)
            prediction = (outputs > 0.5).int().numpy().tolist()
        return prediction