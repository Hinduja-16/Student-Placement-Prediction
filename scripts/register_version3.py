from mlflow import MlflowClient

client = MlflowClient()

model_name = "Student_Placement_LogisticRegression"

model_id = "m-dc29ce9351fd442b9f2ad1de53d25b94"

version = client.create_model_version(
    name=model_name,
    source=f"models:/{model_id}",
    run_id="fad2187d1d1d4b14ac3e9c9bb238ca81"
)

print("\nModel Version 3 registered successfully!")
print("Model Name:", model_name)
print("Version:", version.version)
print("Run ID:", version.run_id)
print("Model ID:", model_id)