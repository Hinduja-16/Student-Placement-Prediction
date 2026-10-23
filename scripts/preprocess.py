import pandas as pd

input_file = "data/raw/student_placement_raw.csv"
output_file = "data/processed/student_placement_processed.csv"

df = pd.read_csv(input_file)

print("Raw data shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert target column into numbers
df["Placement"] = df["Placement"].map({
    "Yes": 1,
    "No": 0
})

# Save processed data
df.to_csv(output_file, index=False)

print("\nProcessed data shape:", df.shape)
print("Processed dataset saved successfully!")