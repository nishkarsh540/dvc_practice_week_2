import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("🚀 Starting model training pipeline...")

# === Load dataset ===
DATA_PATH = "data/iris.csv"
print(f"📂 Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# === Prepare features and target ===
print("🔧 Preparing features and target columns...")
X = df.drop(columns=["species"])
y = df["species"]

# === Split the data ===
print("✂️ Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train a Decision Tree model ===
print("🎯 Training Decision Tree Classifier...")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# === Evaluate model ===
print("📊 Evaluating model on test data...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# === Save the model ===
ARTIFACTS_DIR = "./artifacts"
print(f"💾 Saving model to: {ARTIFACTS_DIR}")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
joblib.dump(clf, MODEL_PATH)

print(f"🎉 Model saved successfully at: {MODEL_PATH}")
