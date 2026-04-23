import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# load CSV
df = pd.read_csv("keypoints.csv")
print(f"Total samples: {len(df)}")
print(f"Poses: {df['label'].value_counts().to_dict()}")

# split features and label
X = df.drop("label", axis=1).values
y = df["label"].values

# split into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# train Random Forest
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.1f}%")
print("\nDetailed results:")
print(classification_report(y_test, y_pred))

# save model
with open("pose_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to pose_model.pkl")