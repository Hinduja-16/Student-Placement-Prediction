import mlflow
from mlflow import MlflowClient

run_id = "51fc95bbbe3c42d89e06be01435dd31c"

model_name = "Student_Placement_LogisticRegression"

model_uri = f"runs:/{run_id}/model"

client = MlflowClient()

# Create registered model if it does not exist
try:
    client.create_registered_model(model_name)
    print("Registered model created.")
except Exception:
    print("Registered model already exists.")

# Create model version
version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

print("\nModel registered successfully!")
print("Model Name:", model_name)
print("Version:", version.version)
print("Run ID:", run_id)