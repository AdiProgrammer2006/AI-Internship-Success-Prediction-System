import pandas as pd  # data manipulation and analysis library
from sklearn.model_selection import train_test_split, cross_val_score  # train_test_split: splits data into training and testing sets | cross_val_score: evaluates model using cross validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # RandomForestClassifier: model using multiple decision trees | GradientBoostingClassifier: model that builds trees sequentially to reduce errors
from sklearn.linear_model import LogisticRegression  # simple classification model that predicts probability of an outcome
from sklearn.metrics import classification_report, confusion_matrix  # classification_report: shows precision, recall, f1 per class | confusion_matrix: table showing correct vs wrong predictions
from sklearn.preprocessing import StandardScaler  # scales features to have mean=0 and standard deviation=1 so all features are on the same scale
from sklearn.pipeline import Pipeline  # chains multiple steps (e.g. scaling + model) into a single object
import joblib  # saves and loads trained models to/from disk
import seaborn as sns  # statistical data visualization library built on matplotlib
import matplotlib.pyplot as plt  # core plotting library for charts and graphs

# Load preprocessed data
df = pd.read_csv(r"data/preprocessed_data.csv")

print("Dataset shape:", df.shape)
print("Placement distribution:\n", df["placement_status"].value_counts())

# Drop salary as it leaks placement status
X = df.drop(["placement_status"], axis=1)
y = df["placement_status"]

# Split with stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define Models 
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=50,     # high value = less overfitting on large datasets
        min_samples_split=100,   # high value = less overfitting on large datasets
        max_features="sqrt",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,             # shallow trees generalize better
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
}

# Compare All Models 
print("\n--- Cross Validation Scores ---")
best_model = None
best_cv_score = 0
best_name = ""

for name, m in models.items():
    cv_scores = cross_val_score(m, X, y, cv=5, scoring="accuracy")
    print(f"{name:25s} -> CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    if cv_scores.mean() > best_cv_score:
        best_cv_score = cv_scores.mean()
        best_model = m
        best_name = name

#  Train Best Model 
print(f"\nBest Model: {best_name}")
best_model.fit(X_train, y_train)

train_acc = best_model.score(X_train, y_train)
test_acc  = best_model.score(X_test, y_test)
gap       = train_acc - test_acc

print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f}")
print(f"Gap               : {gap:.4f}", end="  ")
print("✓ Healthy" if gap < 0.10 else "✗ Still overfitting")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# Confusion Matrix 
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
plt.show()

# Feature Importance
if hasattr(best_model, "feature_importances_"):
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title(f"Top 15 Features - {best_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    plt.show()

elif hasattr(best_model, "coef_"):
    # For Logistic Regression
    feature_importances = pd.Series(
        abs(best_model.coef_[0]), index=X.columns
    )
    top_features = feature_importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title(f"Top 15 Features - {best_name}")
    plt.xlabel("Coefficient Magnitude")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    plt.show()

joblib.dump(best_model, r"model/placement_model.pkl")
print(f"\n{best_name} saved to model/placement_model.pkl")