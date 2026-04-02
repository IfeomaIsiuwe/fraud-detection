"""
API for serving fraud detection model predictions.
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from data_loader import load_engineered, TARGET, NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES


# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_pipeline.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run notebooks first.")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud detection predictions",
    version="1.0.0"
)


class TransactionFeatures(BaseModel):
    """Input features for a single transaction."""
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    merchant_category: str = Field(..., description="Merchant category")
    foreign_transaction: int = Field(..., ge=0, le=1, description="1 if foreign transaction, 0 otherwise")
    location_mismatch: int = Field(..., ge=0, le=1, description="1 if location mismatch, 0 otherwise")
    device_trust_score: float = Field(..., ge=0, le=100, description="Device trust score (0-100)")
    velocity_last_24h: int = Field(..., ge=0, description="Number of transactions in last 24h")
    cardholder_age: int = Field(..., ge=18, le=100, description="Cardholder age")


class PredictionResponse(BaseModel):
    """Prediction response."""
    fraud_probability: float = Field(..., ge=0, le=1, description="Predicted probability of fraud")
    fraud_prediction: int = Field(..., ge=0, le=1, description="Binary prediction (1=fraud, 0=legitimate)")
    threshold_used: float = Field(..., ge=0, le=1, description="Threshold used for binary prediction")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[TransactionFeatures] = Field(..., description="List of transactions to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


def preprocess_transaction(transaction: TransactionFeatures) -> pd.DataFrame:
    """Convert transaction to DataFrame and apply feature engineering."""
    # Convert to DataFrame
    df = pd.DataFrame([transaction.dict()])

    # Apply feature engineering (same as in data_loader.py)
    df["amount_log"] = df["amount"].apply(lambda x: pd.np.log1p(x) if hasattr(pd, 'np') else __import__('numpy').log1p(x))
    df["amount_x_velocity"] = df["amount"] * df["velocity_last_24h"]
    df["high_risk_hour"] = df["transaction_hour"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)
    df["velocity_bin"] = pd.qcut(df["velocity_last_24h"], q=4, labels=[0, 1, 2, 3]).astype(int)

    return df


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Fraud Detection API is running", "status": "healthy"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionFeatures,
    threshold: Optional[float] = 0.5
):
    """
    Predict fraud probability for a single transaction.

    - **threshold**: Classification threshold (default: 0.5)
    """
    try:
        # Validate threshold
        if not (0 < threshold < 1):
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

        # Preprocess
        df = preprocess_transaction(transaction)

        # Predict
        proba = model.predict_proba(df)[:, 1][0]
        prediction = int(proba >= threshold)

        return PredictionResponse(
            fraud_probability=round(float(proba), 4),
            fraud_prediction=prediction,
            threshold_used=threshold
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: BatchPredictionRequest,
    threshold: Optional[float] = 0.5
):
    """
    Predict fraud for multiple transactions.

    - **threshold**: Classification threshold (default: 0.5)
    """
    try:
        # Validate threshold
        if not (0 < threshold < 1):
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

        # Preprocess all transactions
        dfs = [preprocess_transaction(tx) for tx in request.transactions]
        df_batch = pd.concat(dfs, ignore_index=True)

        # Predict
        probas = model.predict_proba(df_batch)[:, 1]
        predictions = (probas >= threshold).astype(int)

        # Build response
        response_predictions = [
            PredictionResponse(
                fraud_probability=round(float(proba), 4),
                fraud_prediction=int(pred),
                threshold_used=threshold
            )
            for proba, pred in zip(probas, predictions)
        ]

        return BatchPredictionResponse(predictions=response_predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)