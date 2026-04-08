import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "placement_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "normalized_placement_data.csv")
 
FEATURE_COLS = [
    "Academic_Score",    # CGPA mapped to 0–10
    "DSA_Skill",         # self-rated DSA ability 1–10
    "Project_Quality",   # project depth/impact 1–10
    "Experience_Score",  # internship / work-ex score 1–10
    "OpenSource_Value",  # open-source contributions 1 or 10
    "Soft_Skills",       # communication/soft-skills 1–10
    "Tech_Stack_Score",  # breadth of tech stack 1–10
]
 
 
def train_and_save() -> dict:
    """Train the regression model and persist it. Returns evaluation metrics."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["Readiness_Score"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
 
    preds = model.predict(X_test)
    metrics = {
        "mse": round(mean_squared_error(y_test, preds), 4),
        "r2":  round(r2_score(y_test, preds), 4),
    }
 
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
 
    print(f"[Regression] Model saved → MSE={metrics['mse']}  R²={metrics['r2']}")
    return metrics
 
 
def load_model():
    """Load persisted model, training it first if not found."""
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
 
 
def predict(profile: dict) -> float:
    """
    Predict placement readiness score (0–100) from a profile dict.
 
    Expected keys (matching FEATURE_COLS):
        Academic_Score, DSA_Skill, Project_Quality, Experience_Score,
        OpenSource_Value, Soft_Skills, Tech_Stack_Score
    """
    model = load_model()
    row = pd.DataFrame([{col: profile.get(col, 5) for col in FEATURE_COLS}])
    score = float(model.predict(row)[0])
    return round(min(max(score, 0), 100), 2)
 
 
if __name__ == "__main__":
    metrics = train_and_save()
    print(f"Training complete: {metrics}")
 
    # quick sanity check
    sample = {
        "Academic_Score": 8.5,
        "DSA_Skill": 7,
        "Project_Quality": 8,
        "Experience_Score": 6,
        "OpenSource_Value": 10,
        "Soft_Skills": 8,
        "Tech_Stack_Score": 7,
    }
    print(f"Sample prediction: {predict(sample)}")