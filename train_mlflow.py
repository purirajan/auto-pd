import hashlib
import platform
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_source import get_history_and_new_apps
from validate import build_schema

# Regularization strength comes from the command line, e.g. python train_mlflow.py 0.1
C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MODEL_NAME = "auto_pd"

NUM_FEATURES = ["duration", "credit_amount", "installment_commitment",
                "age", "existing_credits"]
CAT_FEATURES = ["purpose", "checking_status", "credit_history",
                "savings_status", "employment", "housing"]
FEATURES = NUM_FEATURES + CAT_FEATURES

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("auto-pd")

history, _ = get_history_and_new_apps()
X, y = history[FEATURES], history["default_flag"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
data_hash = hashlib.sha256(
    pd.util.hash_pandas_object(history, index=True).values
).hexdigest()[:16]

model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])),
    ("clf", LogisticRegression(C=C, max_iter=1000)),
])

with mlflow.start_run(run_name=f"logreg_C={C}") as run:
    model.fit(X_train, y_train)
    pd_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pd_test)

    # Lab notebook: settings, results, and context
    mlflow.log_params({"model_type": "logistic_regression", "C": C,
                       "test_size": 0.3, "random_state": 42})
    mlflow.log_metrics({"test_auc": auc, "test_gini": 2 * auc - 1,
                        "mean_predicted_pd": float(pd_test.mean()),
                        "observed_default_rate": float(y_test.mean())})
    mlflow.set_tags({"data_source": "OpenML credit-g v1, car loans only",
                     "data_hash": data_hash,
                     "python_version": platform.python_version(),
                     "sklearn_version": sklearn.__version__})

    # The data contract travels with the run
    mlflow.log_dict(build_schema(X_train, NUM_FEATURES, CAT_FEATURES),
                    "input_schema.json")

    # Package the model and register a new version in the recipe book
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=infer_signature(X_train, model.predict(X_train)),
        input_example=X_train.head(3),
        registered_model_name=MODEL_NAME,
    )

    print(f"C={C}  AUC={auc:.3f}  run_id={run.info.run_id}")