import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. Data Handling
# ============================================================

# Load the mushroom dataset
df = pd.read_csv("data/Mushroom data.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst few rows:")
print(df.head())
print("\nLabel distribution:")
print(df["Mushroom_quality"].value_counts())

# --- Encode labels as binary (0 = edible, 1 = poisonous) ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Mushroom_quality"])  # e=0, p=1
print("\nLabel mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# --- One-Hot Encode all features ---
X = pd.get_dummies(df.drop("Mushroom_quality", axis=1))
print("\nFeature matrix shape after one-hot encoding:", X.shape)

# Convert to numpy arrays
X = X.values.astype(float)
y = y.astype(float)

# --- Split into L (labeled 10%), U (unlabeled pool 80%), T (test 10%) ---
# First split off the test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Then split the remaining 90% into L (10% of total) and U (80% of total)
# 10/90 ≈ 0.111 of the remaining data goes to L
X_labeled, X_pool, y_labeled, y_pool = train_test_split(
    X_temp, y_temp, test_size=8/9, random_state=42, stratify=y_temp
)

print("\n--- Dataset Split ---")
print(f"Labeled set (L):   {X_labeled.shape[0]} samples ({X_labeled.shape[0]/len(y)*100:.1f}%)")
print(f"Unlabeled pool (U): {X_pool.shape[0]} samples ({X_pool.shape[0]/len(y)*100:.1f}%)")
print(f"Test set (T):       {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")
