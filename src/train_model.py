import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv(r"data/preprocessed_data.csv")

X = df.drop("placement_status", axis=1)
y = df["placement_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model  = RandomForestClassifier()
model.fit(X_train, y_train)

print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

joblib.dump(model, r"model/placement_model.pkl")

print("Model trained and saved successfully.")