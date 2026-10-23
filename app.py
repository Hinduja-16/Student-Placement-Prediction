from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd


# Create FastAPI application
app = FastAPI(
    title="Student Placement Prediction API",
    description="API for predicting student placement using MLflow model",
    version="1.0"
)


# MLflow model

MODEL_URI = "mlruns/1/models/m-dc29ce9351fd442b9f2ad1de53d25b94/artifacts"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Input data structure
class StudentInput(BaseModel):

    Age: int
    CGPA: float
    Attendance: float
    Coding_Score: int
    Aptitude_Score: int
    Communication_Score: int
    Internships: int
    Projects: int
    Backlogs: int
    Academic_Score: float
    Skill_Score: float
    Experience_Score: int
    Has_Backlogs: int


# Health check
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Student_Placement_LogisticRegression",
        "version": "champion"
    }


# Feature information
@app.get("/features")
def features():

    return {
        "features": [
            "Age",
            "CGPA",
            "Attendance",
            "Coding_Score",
            "Aptitude_Score",
            "Communication_Score",
            "Internships",
            "Projects",
            "Backlogs",
            "Academic_Score",
            "Skill_Score",
            "Experience_Score",
            "Has_Backlogs"
        ]
    }


# Prediction endpoint
@app.post("/predict")
def predict(student: StudentInput):

    input_data = pd.DataFrame([student.model_dump()])

    prediction = model.predict(input_data)

    result = prediction[0]

    return {
        "prediction": str(result)
    }