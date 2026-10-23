import mlflow

experiment = mlflow.get_experiment_by_name(
    "Student_Placement_Experiment"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'Logistic Regression'",
    order_by=["start_time DESC"]
)

print("\nLogistic Regression Runs:")
print(
    runs[
        [
            "run_id",
            "status",
            "start_time",
            "metrics.f1_score",
            "metrics.accuracy"
        ]
    ].to_string(index=False)
)