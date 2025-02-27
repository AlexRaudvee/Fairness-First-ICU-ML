import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample DataFrame: Replace with your actual data
df = pd.read_csv("./physionet.org/files/widsdatathon2020/1.0.0/data/cleaned_imputed.csv")  # Load dataset

# Separate features and target
X = df.drop(columns=['hospital_death'])  # Assuming 'target' is your label column
y = df['hospital_death']

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# One-Hot Encode categorical variables
ohe = OneHotEncoder(drop='first', sparse=False)  # drop='first' avoids dummy variable trap
X_encoded = pd.DataFrame(ohe.fit_transform(X[categorical_columns]))

# Restore column names after encoding
X_encoded.columns = ohe.get_feature_names_out(categorical_columns)

# Combine numerical and encoded categorical features
X_final = pd.concat([X_encoded, X[numerical_columns].reset_index(drop=True)], axis=1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Get feature importance from logistic regression coefficients
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg.coef_[0]
})

# Sort by absolute importance
feature_importance['Importance'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display results
print("\nResults of Feature Importance with Logistic Regression:")
print(feature_importance)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})

# Sort and display
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nResults of Feature Importance with Random Forrest: ")
print(feature_importance)
