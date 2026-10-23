from mlflow import MlflowClient

client = MlflowClient()

model_name = "Student_Placement_LogisticRegression"

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version="2"
)

print("Champion alias updated successfully!")
print("Model:", model_name)
print("Alias: champion")
print("Version: 2")