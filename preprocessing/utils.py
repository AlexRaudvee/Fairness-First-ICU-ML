# data wrangling
import pandas as pd

# data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def drop_high_NaN_features(df: pd.DataFrame):
    
    missing_percent = df.isnull().sum() * 100 / len(df)
    columns_to_drop = missing_percent[missing_percent > 50].index
    cleaned_df = df.drop(columns=columns_to_drop)
    
    return cleaned_df, columns_to_drop


def impute_values_for_features(df: pd.DataFrame):
    median_imputer = SimpleImputer(strategy='median')  # We can change this to mean imputer as well
    mode_imputer = SimpleImputer(strategy='most_frequent')

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    imputed_df = df.copy()
    for column in numerical_columns:
        imputed_df[[column]] = median_imputer.fit_transform(imputed_df[[column]])

    for column in categorical_columns:
        imputed_df[[column]] = mode_imputer.fit_transform(imputed_df[[column]])
        
    return imputed_df


def standard_scaling_of_features(df: pd.DataFrame, int_cols: list, float_cols: list) -> pd.DataFrame:
    # Initialize the scaler
    scaler = StandardScaler()

    # Combine the lists but only use those features that are in the DataFrame
    features = [att for att in (int_cols + float_cols) if att in df.columns]
    missing_features = set(int_cols + float_cols) - set(features)
    if missing_features:
        print("These features are missing from the DataFrame:", missing_features)
        
    
    # Fit the scaler to the numeric features and transform them
    df[features] = scaler.fit_transform(df[features])
    return df


def drop_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.5):    
    numeric_data = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_data.corr()
    
    to_drop = set()  # Set to store features to be dropped

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):  # Avoid self-correlation
            if (abs(corr_matrix.iloc[i, j]) > threshold) or (abs(corr_matrix.iloc[i, j]) < -threshold):
                feature_to_drop = corr_matrix.columns[j]  # Drop the second feature in the pair
                to_drop.add(feature_to_drop)
                
    to_drop.add("readmission_status") # remove the full zeros feature
    
    return df.drop(columns=to_drop, errors='ignore')


def print_nan_info(df):

    nan_counts = df.isna().sum()  # Count NaN values per column
    nan_percentage = (nan_counts / len(df)) * 100  # Convert to percentage
    
    nan_df = pd.DataFrame({'Missing Values': nan_counts, 'Percentage': nan_percentage})
    nan_df = nan_df[nan_df['Missing Values'] > 0]  # Filter out columns without NaNs
    nan_df = nan_df.sort_values(by='Percentage', ascending=False)  # Sort by percentage
    
    if nan_df.empty:
        print("No missing values in the dataset.")
    else:
        print(nan_df)

    
def drop_outliers(df: pd.DataFrame, int_cols: list, float_cols: list) -> pd.DataFrame:
    iso_forest = IsolationForest(contamination=0.05, random_state=69)

    # Combine the lists but only use those features that are in the DataFrame
    features = [att for att in (int_cols + float_cols) if att in df.columns]
    missing_features = set(int_cols + float_cols) - set(features)
    if missing_features:
        print("These features are missing from the DataFrame:", missing_features)
        
    # Initialize and fit the IsolationForest
    # 'contamination' parameter sets the proportion of expected outliers
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(df[features])

    # Predict: 1 for inliers, -1 for outliers
    df['anomaly'] = iso_forest.predict(df[features])

    # Keep only the inliers
    df_clean = df[df['anomaly'] == 1].drop('anomaly', axis=1)

    return df_clean
