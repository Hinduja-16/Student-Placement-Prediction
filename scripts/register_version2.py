import mlflow
from mlflow import MlflowClient

run_id = "71fb3051df6d44d88a077a8c61441e58"

model_name = "Student_Placement_LogisticRegression"

model_uri = f"runs:/{run_id}/model"

client = MlflowClient()

version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

print("\nModel Version 2 registered successfully!")
print("Model Name:", model_name)
print("Version:", version.version)
print("Run ID:", run_id)