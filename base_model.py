import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and concatenate NASA battery data
file_paths = glob('/mnt/data/CSE147_Team9_Project/data/nasa_cleaned/*.csv')
df_list = [pd.read_csv(path) for path in file_paths]
df = pd.concat(df_list, ignore_index=True)

# 2. Select features and target
# Assuming columns: 'cycle_number', 'temperature', 'current', 'voltage', 'capacity'
features = ['cycle_number', 'temperature', 'current', 'voltage']
target = 'capacity'

X = df[features]
y = df[target]

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build pipeline with scaling and baseline Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Baseline Linear Regression Results:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  R^2 Score: {r2:.4f}")
