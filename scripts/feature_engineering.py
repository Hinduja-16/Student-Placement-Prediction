import pandas as pd

input_file = "data/processed/student_placement_processed.csv"
output_file = "data/features/student_placement_features.csv"

df = pd.read_csv(input_file)

# Create Academic Score
df["Academic_Score"] = (
    df["CGPA"] * 10
    + df["Attendance"] * 0.2
)

# Create Skill Score
df["Skill_Score"] = (
    df["Coding_Score"]
    + df["Aptitude_Score"]
    + df["Communication_Score"]
) / 3

# Create Experience Score
df["Experience_Score"] = (
    df["Internships"] * 2
    + df["Projects"]
)

# Create Backlog indicator
df["Has_Backlogs"] = (df["Backlogs"] > 0).astype(int)

# Save feature-engineered data
df.to_csv(output_file, index=False)

print("Feature engineering completed!")
print("Final shape:", df.shape)

print("\nNew features:")
print("Academic_Score")
print("Skill_Score")
print("Experience_Score")
print("Has_Backlogs")