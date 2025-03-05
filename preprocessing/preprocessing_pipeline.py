# data wrangling
import pandas as pd

# data preprocessing
from sklearn.model_selection import train_test_split

# custom imports
from utils import *

# PREPROCESSING PIPELINE

df: pd.DataFrame = pd.read_csv('physionet.org/files/widsdatathon2020/1.0.0/data/training_v2.csv')
df_: pd.DataFrame = pd.read_csv('physionet.org/files/widsdatathon2020/1.0.0/data/WiDS_Datathon_2020_Dictionary.csv')

BINARY_ATTS = df_[df_['Data Type'].str.lower() == 'binary']['Variable Name'].tolist()
print(f"Binary atts : {BINARY_ATTS}")
STRING_ATTS = df_[df_['Data Type'].str.lower() == 'string']['Variable Name'].tolist()
print(f"String atts : {STRING_ATTS}")
INT_ATTS = df_[df_['Data Type'].str.lower() == 'integer']['Variable Name'].tolist()
print(f"Integer atts : {INT_ATTS}")
FLOAT_ATTS = df_[df_['Data Type'].str.lower() == 'numeric']['Variable Name'].tolist()
FLOAT_ATTS.remove('pred')
print(f"Float atts : {FLOAT_ATTS}")

# drop the id columns
df = df.drop(columns=["encounter_id", "patient_id", "hospital_id", "icu_id", "icu_stay_type", "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"])

print("\nDrop features with high NANs: ")
print(f"Feature Number Before: {len(df.columns)}")
df, columns_to_drop = drop_high_NaN_features(df)
print(f"Feature Number After: {len(df.columns)}")

print("\nImpute Remaining Missing Values: ")
print(f"Total NANs Before: \n{df.isnull().sum().sort_values(ascending=False)}")
df = impute_values_for_features(df)
print(f"Total NANs After: \n{df.isnull().sum().sort_values(ascending=False)}")

print("\nFilter out outliers: \n")
print(f"Shape of the Dataframe before: {df.shape}")
df = drop_outliers(df, int_cols=INT_ATTS, float_cols=FLOAT_ATTS)
print(f"Shape of the Dataframe after: {df.shape}")

print("\nScale The Data For Better Visualization: ")
print(f"Description Before: \n{df.select_dtypes(include=['int64', 'float64']).describe(include='all')}")
df = standard_scaling_of_features(df, int_cols=INT_ATTS, float_cols=FLOAT_ATTS)
print(f"Description After: \n{df.select_dtypes(include=['int64', 'float64']).describe(include='all')}")

print("\nDrop Highly Correlated Features: ")
print(f"Feature Number Before: {len(df.columns)}")
df = drop_highly_correlated_features(df)
print(f"Feature Number After: {len(df.columns)}")

print("\n Save the DF: ")
df.to_csv("physionet.org/files/widsdatathon2020/1.0.0/data/cleaned_imputed.csv")
print(f"Saved at: {'physionet.org/files/widsdatathon2020/1.0.0/data/'}")