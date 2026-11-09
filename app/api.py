from pathlib import Path
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
from loguru import logger
import json

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_sentimento.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="API de Análise de Sentimento", version="1.0")

class Entrada(BaseModel):
    texto: str = Field(min_length = 3, max_length = 200, example="Eu amo este produto!")

class Saida(BaseModel):
    sentimento: int
    confianca: float
    versao_modelo: str

@app.get("/health", summary="Verifica a saúde da API")
def health():
    return {"status": "ok", "versao_modelo": "1.0"}

@app.post("/predict", response_model=Saida, summary="Prediz o sentimento do texto fornecido")
def predict(entrada: Entrada) -> Saida:
    texto_processado = entrada.texto.lower()
    
    # Realiza a predição usando o modelo carregado
    sentimento = model.predict([texto_processado])[0]
    confianca = max(model.predict_proba([texto_processado])[0])
    
    logger.info(f"Predição realizada: Sentimento={sentimento}, Confiança={confianca:.2f}")
    
    return Saida(sentimento=sentimento, confianca=confianca, versao_modelo="1.0")