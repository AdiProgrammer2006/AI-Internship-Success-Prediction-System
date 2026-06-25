import pandas as pd

# Load dataset
df = pd.read_csv(r"data/raw_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Show data
print(df.head())
print(df.columns)


df = df.dropna()


binary_map = {
    "Yes": 1,
    "No": 0
}

df["Internships(Y/N)"] = df["Internships(Y/N)"].map(binary_map)
df["Training(Y/N)"] = df["Training(Y/N)"].map(binary_map)
df["Innovative Project(Y/N)"] = df["Innovative Project(Y/N)"].map(binary_map)
df["Technical Course(Y/N)"] = df["Technical Course(Y/N)"].map(binary_map)
df["Backlog in 5th sem"] = df["Backlog in 5th sem"].map(binary_map)

df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})


df["Placement(Y/N)?"] = df["Placement(Y/N)?"].map({
    "Placed": 1,
    "Not Placed": 0
})


df = pd.get_dummies(df, columns=["10th board", "12th board", "Stream"])



df.to_csv(r"data/preprocessed_data.csv", index=False)

print("Preprocessing completed successfully!")