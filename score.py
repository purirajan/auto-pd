import json
import sys
from datetime import datetime

import joblib
import pandas as pd

from data_source import get_history_and_new_apps
from validate import validate

model = joblib.load("artifacts/pd_model.joblib")
with open("artifacts/model_metadata.json") as f:
    meta = json.load(f)
with open("artifacts/input_schema.json") as f:
    schema = json.load(f)

_, new_apps = get_history_and_new_apps()

# --- Validation gate ---
errors, warnings = validate(new_apps, schema)
report = {
    "checked_at": datetime.now().isoformat(timespec="seconds"),
    "model_version": meta["model_version"],
    "rows": len(new_apps),
    "status": "FAILED" if errors else "PASSED",
    "errors": errors,
    "warnings": warnings,
}
with open("output/validation_report.json", "w") as f:
    json.dump(report, f, indent=2)

for w in warnings:
    print(f"WARNING: {w}")

if errors:
    print("VALIDATION FAILED. Batch was NOT scored:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

# --- Scoring (only reached if validation passed) ---
pd_scores = model.predict_proba(new_apps[meta["features"]])[:, 1]

out = pd.DataFrame({
    "loan_id": new_apps["loan_id"].values,
    "purpose": new_apps["purpose"].values,
    "pd": pd_scores.round(4),
    "model_version": meta["model_version"],
    "scored_at": datetime.now().isoformat(timespec="seconds"),
})
out.to_csv("output/scored_applications.csv", index=False)
print(f"Validation passed. Scored {len(out)} applications with model v{meta['model_version']}")