import pandas as pd
from glob import glob
from sklearn.ensemble import RandomForestRegressor

# 1. Load and concatenate NASA battery data
file_paths = glob('/mnt/data/CSE147_Team9_Project/data/nasa_cleaned/*.csv')
df_list = [pd.read_csv(path) for path in file_paths]
df = pd.concat(df_list, ignore_index=True)

# 2. Define features and target
features = ['cycle_number', 'temperature', 'current', 'voltage']
target = 'capacity'

X = df[features]
y = df[target]

# 3. Compute Pearson correlations with the target
corr_df = pd.DataFrame({
    'feature': features,
    'correlation': [X[col].corr(y) for col in features]
})
corr_df['abs_correlation'] = corr_df['correlation'].abs()
corr_df = corr_df.sort_values('abs_correlation', ascending=False)

# 4. Fit a Random Forest for feature importances
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

imp_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

# 5. Display results
import pandas as pd
from ace_tools import display_dataframe_to_user

display_dataframe_to_user(name="Feature Correlations with Capacity", dataframe=corr_df.reset_index(drop=True))
display_dataframe_to_user(name="Random Forest Feature Importances", dataframe=imp_df.reset_index(drop=True))
