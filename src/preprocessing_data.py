import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()
df = df.dropna()
df = df.drop(columns =["student_id"])

# Show data
print(df.head())
print(df.columns)


binary_map = {
    "Yes": 1,
    "No": 0
}

df["volunteer_experience"] = df["volunteer_experience"].map(binary_map)

#encoding gender
df["gender"]= df["gender"].map({
    "Male": 1,
    "Female": 0
})

#encode target column
df["placement_status"] = df["placement_status"].map(
    {
        "Placed":1,
        "Not Placed":0
    })

#oneshot encode categorical columns
df = pd.get_dummies(
    df,
    columns = ["branch", "college_tier"],
    dtype = int
)



#save peprocessed data into data file
df.to_csv(r"data/preprocessed_data.csv", index=False)
print("Preprocessing completed successfully!")