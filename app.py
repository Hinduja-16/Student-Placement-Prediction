
import logging
import sys
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel
import mlflow
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")


# Create FastAPI application
app = FastAPI(
    title="Student Placement Prediction API",
    description="API for predicting student placement using MLflow model",
    version="1.0"
)


# Load MLflow model
MODEL_URI = "mlruns/1/models/m-dc29ce9351fd442b9f2ad1de53d25b94/artifacts"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("event=model_loaded model_file=%s", MODEL_URI)
except Exception:
    logger.exception("event=model_load_failed")
    raise


# Request and response logging
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        level = (
            logging.WARNING
            if response.status_code >= 400
            else logging.INFO
        )

        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms
        )
        raise


# Validation error logging
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = [
        ".".join(str(part) for part in error.get("loc", []))
        for error in exc.errors()
    ]

    logger.warning(
        "event=validation_error request_id=%s path=%s field_locations=%s error_count=%s",
        getattr(request.state, "request_id", "unknown"),
        request.url.path,
        fields,
        len(exc.errors())
    )

    return await request_validation_exception_handler(request, exc)


# Student input structure
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


# Safe prediction logging
SAFE_INPUT_FIELDS = set()


@app.post("/predict")
def predict(student: StudentInput, request: Request):
    try:
        payload = student.model_dump()

        safe_input = {
            key: value
            for key, value in payload.items()
            if key in SAFE_INPUT_FIELDS
        }

        logger.info(
            "event=prediction_request request_id=%s feature_names=%s safe_input=%s",
            request.state.request_id,
            sorted(payload.keys()),
            safe_input
        )

        # Preserve existing MLflow prediction logic
        input_data = pd.DataFrame([payload])
        prediction = model.predict(input_data)
        result = prediction[0]

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            request.state.request_id,
            str(result)
        )

        return {
            "prediction": str(result)
        }

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request.state.request_id
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
