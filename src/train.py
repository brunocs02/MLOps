import joblib
import pandas as pd
from pathlib import Path
from loguru import logger

from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "comentarios.csv"
MODELS = ROOT / "models"

def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")

    return df["texto"].tolist(), df["sentimento"].tolist()

def main():

    X, y = load_data(DATA)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with TfidfVectorizer and LogisticRegression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Test Accuracy: {accuracy:.2f}")
    logger.info(f"Test F1 Score: {f1:.2f}")


    joblib.dump(pipeline, MODELS/"model_sentimento.pkl")
    logger.success(f"\nModel saved to {MODELS/'model_sentimento.pkl'}")

if __name__ == "__main__":
    main()