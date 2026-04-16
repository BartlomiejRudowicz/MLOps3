import bentoml
import torch
from lightning_module import BreastCancerLightningModule

# Podaj dokładną ścieżkę do zapisanego pliku .ckpt
CHECKPOINT_PATH = r"C:\Users\rudow\Desktop\MLOps-main\MLOps-main\lightning_logs\version_1\checkpoints\best-model-epoch=15-val_loss=0.0638.ckpt"

def main():
    print(f"Ładowanie modelu Lightning z: {CHECKPOINT_PATH}")
    
    # 1. Wczytanie modelu z pliku checkpointu
    lightning_model = BreastCancerLightningModule.load_from_checkpoint(CHECKPOINT_PATH)
    
    # 2. Przełączenie w tryb ewaluacji (wyłącza np. Dropout podczas inferencji)
    lightning_model.eval()

    # 3. Wyciągnięcie wewnętrznego modelu PyTorch (zmiennej self.model z Twojego kodu)
    pytorch_model = lightning_model.model

    # 4. Zapisanie modelu do wbudowanego magazynu BentoML
    # "breast_cancer_mlp" to nazwa, której będziemy używać w kodzie serwera
    saved_model = bentoml.pytorch.save_model("breast_cancer_mlp", pytorch_model)
    
    print(f"\nSukces! Model zapisany w BentoML: {saved_model.tag}")

if __name__ == "__main__":
    main()