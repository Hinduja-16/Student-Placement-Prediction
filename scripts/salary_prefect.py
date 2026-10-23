from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib


@task
def load_data():
    print("Loading salary dataset...")
    return pd.read_csv("data/salary_dataset.csv")


@task
def preprocess_data(data):
    print("Preprocessing data...")

    data = pd.get_dummies(
        data,
        columns=["EducationLevel", "JobRole"],
        dtype=int
    )

    return data


@task
def train_model(data):
    print("Training salary prediction model...")

    X = data.drop("Salary", axis=1)
    y = data["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)

    return model, score


@task
def evaluate_model(result):
    model, score = result

    print("Model R² Score:", score)

    if score >= 0.80:
        return model, score, True
    else:
        return model, score, False


@task
def save_model(result):
    model, score, accepted = result

    if accepted:
        joblib.dump(model, "models/salary_model.pkl")
        print("Model saved successfully!")
    else:
        print("Model rejected because R² score is low.")


@flow(name="Salary Prediction Workflow", log_prints=True)
def salary_workflow():

    data = load_data()

    processed_data = preprocess_data(data)

    result = train_model(processed_data)

    evaluated_result = evaluate_model(result)

    save_model(evaluated_result)


if __name__ == "__main__":
    salary_workflow()