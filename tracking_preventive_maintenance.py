import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'machine_id': np.random.randint(1, 21, n_samples),
        'operating_hours': np.random.randint(50, 5000, n_samples),
        'last_maintenance': np.random.randint(1, 365, n_samples),
        'temperature': np.random.uniform(20, 100, n_samples),
        'vibration': np.random.uniform(0.1, 5.0, n_samples),
        'oil_quality': np.random.uniform(0.1, 1.0, n_samples),
        'failure_risk': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

# Load dataset
df = generate_data()
print(df.head())

# Splitting the dataset
X = df[['operating_hours', 'last_maintenance', 'temperature', 'vibration', 'oil_quality']]
y = df['failure_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
feature_importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Predicting Maintenance Needs")
plt.show()
