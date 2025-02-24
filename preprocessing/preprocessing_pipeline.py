import time, matplotlib, sklearn
# visualizatoin 
import matplotlib.pyplot as plt 
import seaborn as sns
# data wrangling
import pandas as pd
import numpy as np 
# data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from utils import *

# PREPROCESSING PIPELINE

df: pd.DataFrame = pd.read_csv('physionet.org/files/widsdatathon2020/1.0.0/data/training_v2.csv')

print("\nDrop features with high NANs: ")
print(f"Feature Number Before: {len(df.columns)}")
df = drop_high_NaN_features(df)
print(f"Feature Number After: {len(df.columns)}")

print("\nImpute Remaining Missing Values: ")
print(f"Total NANs Before: \n{df.isnull().sum().sort_values(ascending=False)}")
df = impute_values_for_features(df)
print(f"Total NANs After: \n{df.isnull().sum().sort_values(ascending=False)}")

print("\nScale The Data For Better Visualization: ")
print(f"Description Before: \n{df.select_dtypes(include=['int64', 'float64']).describe(include='all')}")
df = robust_scaling_of_features(df)
print(f"Description After: \n{df.select_dtypes(include=['int64', 'float64']).describe(include='all')}")

print("\nDrop Highly Correlated Features: ")
print(f"Feature Number Before: {len(df.columns)}")
df = drop_highly_correlated_features(df)
print(f"Feature Number After: {len(df.columns)}")

print("\nSave DF: ")
df.to_csv("physionet.org/files/widsdatathon2020/1.0.0/data/cleaned_imputed_training.csv")
print(f"Saved at: {'physionet.org/files/widsdatathon2020/1.0.0/data/cleaned_imputed_training.csv'}")