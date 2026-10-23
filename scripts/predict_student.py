import mlflow
import pandas as pd

# Champion model URI
model_uri = "models:/Student_Placement_LogisticRegression@champion"

# Load champion model
model = mlflow.sklearn.load_model(model_uri)

print("Champion model loaded successfully!")


# Student input
student = pd.DataFrame([{
    "Age": 22,
    "CGPA": 8.8,
    "Attendance": 90.0,
    "Coding_Score": 85,
    "Aptitude_Score": 80,
    "Communication_Score": 82,
    "Internships": 2,
    "Projects": 3,
    "Backlogs": 0,
    "Academic_Score": 106.0,
    "Skill_Score": 82.333333,
    "Experience_Score": 7,
    "Has_Backlogs": 0
}])


# Make prediction
prediction = model.predict(student)[0]

print("\nStudent Input:")
print(student)

print("\nPrediction:")

if prediction == 1:
    print("Placement: Yes")
else:
    print("Placement: No")