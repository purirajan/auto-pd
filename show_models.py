import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

champion = client.get_model_version_by_alias("auto_pd", "champion").version
versions = client.search_model_versions("name='auto_pd'")

for mv in sorted(versions, key=lambda m: int(m.version)):
    run = client.get_run(mv.run_id)
    c = run.data.params["C"]
    auc = run.data.metrics["test_auc"]
    marker = "  <- champion" if mv.version == champion else ""
    print(f"v{mv.version}   C={c:<6}  AUC={auc:.3f}{marker}")