import joblib
from loguru import logger
from pathlib import Path
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_sentimento.pkl"

lista = ["Amei o produto", "Atendimento desagradável", "Chegou antes do prazo", "Serviço meia-boca"]

def main():

    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")

    predictions = model.predict(lista)
    for text, pred in zip(lista, predictions): 
        logger.info(f"Comentário: {text} | Previsão: {pred}")

if __name__ == "__main__":
    main()  