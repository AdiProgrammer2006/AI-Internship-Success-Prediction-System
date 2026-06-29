import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import numpy as np

df = pd.read_csv(r"data/preprocessed_data.csv")
X = df.drop(["placement_status"], axis=1)
y = df["placement_status"]

# ─────────────────────────────────────────
# 1. Class Distribution Bar Chart
# ─────────────────────────────────────────
plt.figure(figsize=(6, 4))
counts = y.value_counts()
sns.barplot(x=["Not Placed", "Placed"], y=counts.values,
            hue=["Not Placed", "Placed"], palette="Blues_d", legend=False)
for i, v in enumerate(counts.values):
    plt.text(i, v + 300, str(v), ha="center", fontweight="bold")
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/class_distribution.png")
plt.show()
print("✓ Class distribution saved")


plt.figure(figsize=(20, 16))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 6})
plt.title("Full Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("data/heatmap.png")
plt.show()
print("✓ Heatmap saved")

# ─────────────────────────────────────────
# 3. Top 15 Feature Correlations with placement_status
# ─────────────────────────────────────────
correlations = df.corr(numeric_only=True)["placement_status"].drop("placement_status")
correlations = correlations.abs().sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.values, y=correlations.index,
            hue=correlations.index, palette="coolwarm", legend=False)
plt.title("Top 15 Feature Correlations with Placement Status")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("data/feature_correlation.png")
plt.show()
print("✓ Feature correlation saved")

# ─────────────────────────────────────────
# 4. Box Plots — Feature Distributions by Placement
# ─────────────────────────────────────────
top_features = ["coding_skill_score", "mock_interview_score",
                "aptitude_score", "communication_skill_score",
                "logical_reasoning_score", "leadership_score"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    sns.boxplot(x=df["placement_status"], y=df[feature],
                hue=df["placement_status"], palette="Blues",
                legend=False, ax=axes[i])
    axes[i].set_title(f"{feature} by Placement Status")
    axes[i].set_xticklabels(["Not Placed", "Placed"])
    axes[i].set_xlabel("")

plt.suptitle("Feature Distributions by Placement Status", fontsize=14)
plt.tight_layout()
plt.savefig("data/boxplots.png")
plt.show()
print("✓ Box plots saved")

# ─────────────────────────────────────────
# 5. CV Accuracy Comparison Bar Chart
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="saga", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8,
                                            min_samples_leaf=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                    learning_rate=0.05, random_state=42)
}

cv_means = {}
cv_stds  = {}
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="accuracy")
    cv_means[name] = scores.mean()
    cv_stds[name]  = scores.std()
    print(f"{name}: {scores.mean():.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(cv_means.keys(), cv_means.values(),
               yerr=cv_stds.values(), capsize=5,
               color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylim(0.75, 0.92)
for bar, val in zip(bars, cv_means.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", fontweight="bold")
plt.title("Cross Validation Accuracy Comparison")
plt.ylabel("CV Accuracy")
plt.tight_layout()
plt.savefig("data/cv_comparison.png")
plt.show()
print("✓ CV comparison saved")

# ─────────────────────────────────────────
# 6. Confusion Matrix (reload saved model)
# ─────────────────────────────────────────
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed", "Placed"],
            yticklabels=["Not Placed", "Placed"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
plt.show()
print("✓ Confusion matrix saved")

# ─────────────────────────────────────────
# 7. Accuracy Summary Bar Chart
# ─────────────────────────────────────────
metrics = {
    "Training Accuracy": best_model.score(X_train, y_train),
    "Testing Accuracy" : best_model.score(X_test, y_test),
    "CV Score"         : 0.8663
}

plt.figure(figsize=(7, 4))
bars = plt.bar(metrics.keys(), metrics.values(),
               color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylim(0.80, 0.90)
for bar, val in zip(bars, metrics.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", fontweight="bold")
plt.title("Model Accuracy Summary")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("data/accuracy_summary.png")
plt.show()
print("✓ Accuracy summary saved")

print("\n✅ All visuals generated and saved to data/ folder!")