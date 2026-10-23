from mlflow import MlflowClient

client = MlflowClient()

model_name = "Student_Placement_LogisticRegression"

model_version = client.get_model_version_by_alias(
    model_name,
    "champion"
)

print("Champion Model Verification")
print("---------------------------")
print("Model Name:", model_version.name)
print("Version:", model_version.version)
print("Run ID:", model_version.run_id)
print("Alias: champion")