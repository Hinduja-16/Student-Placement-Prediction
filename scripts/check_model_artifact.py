import mlflow
from mlflow import MlflowClient

run_id = "71fb3051df6d44d88a077a8c61441e58"

client = MlflowClient()

print("Run ID:")
print(run_id)

print("\nRun Artifacts:")

artifacts = client.list_artifacts(run_id)

if not artifacts:
    print("No run artifacts found.")
else:
    for item in artifacts:
        print(
            "Name:", item.path,
            "| Is directory:", item.is_dir
        )

print("\nLogged Models:")

try:
    models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run_id}'"
    )

    for model in models:
        print("Model ID:", model.model_id)
        print("Name:", model.name)
        print("Status:", model.status)

except Exception as e:
    print("Could not list logged models:")
    print(e)