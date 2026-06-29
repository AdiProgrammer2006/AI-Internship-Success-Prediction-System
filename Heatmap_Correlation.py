import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/preprocessed_data.csv")


plt.figure(figsize=(18, 14))
sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 7}
)

plt.title("CareerLens AI - Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("data/heatmap.png")  
plt.show()