import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2
import statsmodels.api as sm


RANDOM_SEED = 28
MAX_ITER = 5000
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


def robust_scaling_of_features(df: pd.DataFrame, int_cols: list, float_cols: list) -> pd.DataFrame:
    features = [att for att in (int_cols + float_cols) if att in df.columns]
    df[features] = RobustScaler().fit_transform(df[features])
    return df


def standard_scaling_of_features(df: pd.DataFrame, int_cols: list, float_cols: list) -> pd.DataFrame:
    features = [att for att in (int_cols + float_cols) if att in df.columns]
    df[features] = StandardScaler().fit_transform(df[features])
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
    
    return df.drop(columns=to_drop, errors='ignore'), to_drop


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

def get_path_until_data(s):
    marker = "/data/"
    idx = s.find(marker)
    if idx != -1:
        # Return the substring up to and including "/data/"
        return s[:idx + len(marker)]
    else:
        return None  # or handle the case when marker is not found


def hosmer_lemeshow_test(y_true, y_prob, group=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    n = len(y_true_sorted)
    group_size = n // group

    O1 = np.zeros(group)  # Observed positives
    E1 = np.zeros(group)  # Expected positives
    O0 = np.zeros(group)  # Observed negatives
    E0 = np.zeros(group)  # Expected negatives

    for i in range(group):
        start = i * group_size
        end = (i + 1) * group_size if (i < group - 1) else n
        y_true_chunk = y_true_sorted[start:end]
        y_prob_chunk = y_prob_sorted[start:end]

        O1[i] = np.sum(y_true_chunk)
        E1[i] = np.sum(y_prob_chunk)
        O0[i] = len(y_true_chunk) - O1[i]
        E0[i] = np.sum(1.0 - y_prob_chunk)

    hl_stat = 0.0
    for i in range(group):
        hl_stat += ((O1[i] - E1[i])**2) / (E1[i] + 1e-9)
        hl_stat += ((O0[i] - E0[i])**2) / (E0[i] + 1e-9)

    dof = group - 2
    p_value = 1.0 - chi2.cdf(hl_stat, dof)
    return hl_stat, p_value