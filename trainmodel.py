import pandas as pd
import numpy as np

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
data = pd.read_csv("Testing.csv")

print(data.columns.tolist())
print(data.isnull().sum())
print("Is there any null value in the dataset?", data.isnull().values.any())
print("Number of duplicate rows:", data.duplicated().sum())
print(data.isnull().sum().sum())
X = data.drop("Outcome", axis=1)

y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression(max_iter=1000)


# Training
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
print("Accuracy:", accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# Save
joblib.dump(model,"diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")
