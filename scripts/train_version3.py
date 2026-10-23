import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load feature-engineered data
df = pd.read_csv("data/features/student_placement_features.csv")

print("Data shape:", df.shape)


# Features and target
X = df.drop(columns=["Student_ID", "Placement"])
y = df["Placement"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Set MLflow experiment
mlflow.set_experiment("Student_Placement_Experiment")


# Version 3 model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.25,
        max_iter=1000
    ))
])


with mlflow.start_run(run_name="Logistic Regression Version 3"):

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Log parameters
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("version", "3")
    mlflow.log_param("C", 0.25)
    mlflow.log_param("max_iter", 1000)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # IMPORTANT:
    # Explicitly log the sklearn model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="student_placement_model",
        serialization_format="cloudpickle"
    )

    print("\nVersion 3 Training Complete")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nRun ID:", mlflow.active_run().info.run_id)

    print("\nModel ID:", model_info.model_id)

    print("\nModel URI:", model_info.model_uri)