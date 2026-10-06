import json
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from validate import validate

# --- Load everything ONCE at startup ---
model = joblib.load("artifacts/pd_model.joblib")
with open("artifacts/model_metadata.json") as f:
    meta = json.load(f)
with open("artifacts/input_schema.json") as f:
    schema = json.load(f)

EXAMPLE = {
    "loan_id": 5001,
    "duration": 36,
    "credit_amount": 5000,
    "installment_commitment": 3,
    "age": 35,
    "existing_credits": 1,
    "purpose": "used car",
    "checking_status": "0<=X<200",
    "credit_history": "existing paid",
    "savings_status": "<100",
    "employment": "1<=X<4",
    "housing": "own",
}


# --- The request format: what an "order" must look like ---
class Application(BaseModel):
    loan_id: int
    duration: float
    credit_amount: float
    installment_commitment: float
    age: float
    existing_credits: float
    purpose: str
    checking_status: str
    credit_history: str
    savings_status: str
    employment: str
    housing: str

    model_config = {"json_schema_extra": {"examples": [EXAMPLE]}}


app = FastAPI(title="Auto Loan PD Scoring API", version=meta["model_version"])


@app.get("/health")
def health():
    """Is the service up, and which model is it running?"""
    return {
        "status": "ok",
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
    }


@app.post("/score")
def score(application: Application):
    """Score one auto loan application."""
    row = pd.DataFrame([application.model_dump()])

    # Same validation gate as the batch job
    errors, warnings = validate(row, schema)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    pd_value = float(model.predict_proba(row[meta["features"]])[0, 1])

    return {
        "loan_id": application.loan_id,
        "pd": round(pd_value, 4),
        "model_version": meta["model_version"],
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "warnings": warnings,
    }