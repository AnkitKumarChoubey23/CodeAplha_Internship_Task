# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 📍 Automatically get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Data", "Iris.csv")

# ✅ Load the dataset
try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully from:\n", file_path)
except FileNotFoundError:
    print("❌ File not found at:\n", file_path)
    print("📂 Tip: Make sure 'Data/Iris.csv' exists next to this script")
    exit()

# 🔍 Basic info
print("\n🎯 Dataset Overview:")
print(df.info())
print("\n📊 Summary Statistics:")
print(df.describe())

# 🚫 Remove ID column for analysis
df.drop("Id", axis=1, inplace=True)

# 🌼 Species count
print("\n🌿 Species Count:")
print(df["Species"].value_counts())

# 📊 Pairplot
sns.pairplot(df, hue="Species", diag_kind="kde")
plt.suptitle("🌸 Iris Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# 🔗 Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop("Species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("📊 Feature Correlation Heatmap")
plt.show()

# 📦 Boxplots by species
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
for feature in features:
    plt.figure()
    sns.boxplot(x="Species", y=feature, data=df)
    plt.title(f"📦 {feature} by Species")
    plt.show()