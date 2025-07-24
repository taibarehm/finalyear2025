import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data_stress.csv")  # Adjust path if needed

# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# Prepare features and target
X = data.drop(columns='Stress_Levels')
y = data['Stress_Levels']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)

# Train CatBoost model
model = CatBoostClassifier(random_state=123, verbose=0)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'catboost_stress_model.pkl')
print("Model saved as catboost_stress_model.pkl")