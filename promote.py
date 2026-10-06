import sys

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

version = sys.argv[1]
client.set_registered_model_alias("auto_pd", "champion", version)

mv = client.get_model_version_by_alias("auto_pd", "champion")
print(f"auto_pd v{mv.version} is now champion (from run {mv.run_id})")