import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Student_ID": range(1, n + 1),
    "Age": np.random.randint(20, 25, n),
    "CGPA": np.round(np.random.uniform(5.5, 9.8, n), 2),
    "Attendance": np.round(np.random.uniform(60, 100, n), 2),
    "Coding_Score": np.random.randint(40, 100, n),
    "Aptitude_Score": np.random.randint(40, 100, n),
    "Communication_Score": np.random.randint(40, 100, n),
    "Internships": np.random.randint(0, 4, n),
    "Projects": np.random.randint(0, 6, n),
    "Backlogs": np.random.randint(0, 5, n)
})

score = (
    data["CGPA"] * 10
    + data["Attendance"] * 0.2
    + data["Coding_Score"] * 0.25
    + data["Aptitude_Score"] * 0.20
    + data["Communication_Score"] * 0.15
    + data["Internships"] * 5
    + data["Projects"] * 3
    - data["Backlogs"] * 8
)

data["Placement"] = np.where(score >= 150, "Yes", "No")

data.to_csv(
    "data/raw/student_placement_raw.csv",
    index=False
)

print("Dataset created successfully!")
print("Shape:", data.shape)
print(data.head())