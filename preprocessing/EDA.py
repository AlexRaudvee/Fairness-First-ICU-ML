# visualizatoin 
import matplotlib.pyplot as plt 
import seaborn as sns

# data wrangling
import pandas as pd

# data preprocessing
from sklearn.preprocessing import RobustScaler


df = pd.read_csv('physionet.org/files/widsdatathon2020/1.0.0/data/training_v2.csv')

print("\nGENERAL INFO OF DATASET\n")
print(df.info())

summary_stats = df.describe(include="all")
print(summary_stats)

description = pd.read_csv('physionet.org/files/widsdatathon2020/1.0.0/data/WiDS_Datathon_2020_Dictionary.csv')
print(description[["Variable Name", "Description"]])

print("\nEDA\n")
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop(["encounter_id", "patient_id", "hospital_id"])

num_cols = len(numerical_columns)
cols = 4  
rows = (num_cols + cols - 1) // cols 

plt.figure(figsize=(20, rows * 5)) 
for i, column in enumerate(numerical_columns):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df[column], color='skyblue')  
    plt.title(f'Boxplot of {column}')

plt.tight_layout() 
plt.savefig("assets/boxplot_orig_data.png")
 
 
plt.figure(figsize=(20, rows * 5))  
for i, column in enumerate(numerical_columns):
    plt.subplot(rows, cols, i + 1)
    sns.violinplot(x=df[column], color='skyblue', inner="quartile")  # Violin plot instead of boxplot
    plt.title(f'Violin Plot of {column}')

plt.tight_layout()  
plt.savefig("assets/violinplot_orig_data.png")

numeric_data = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', square=True, cbar_kws={'shrink': .75})
plt.title('Correlation Matrix Heatmap of Numerical Columns')
plt.savefig("assets/corr_matrix_orig_data.png")

print("\nEDA with ROBUST SCALER\n")

df[numerical_columns] = RobustScaler().fit_transform(df[numerical_columns])

plt.figure(figsize=(20, rows * 5)) 
for i, column in enumerate(numerical_columns):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df[column], color='skyblue')  
    plt.title(f'Boxplot of {column}')

plt.tight_layout() 
plt.savefig("assets/boxplot_scaled_data.png")
 
 
plt.figure(figsize=(20, rows * 5))  
for i, column in enumerate(numerical_columns):
    plt.subplot(rows, cols, i + 1)
    sns.violinplot(x=df[column], color='skyblue', inner="quartile")  # Violin plot instead of boxplot
    plt.title(f'Violin Plot of {column}')

plt.tight_layout()  
plt.savefig("assets/violinplot_scaled_data.png")

numeric_data = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', square=True, cbar_kws={'shrink': .75})
plt.title('Correlation Matrix Heatmap of Numerical Columns')
plt.savefig("assets/corr_matrix_scaled_data.png")

print("\nEND\n")
print("###########################")
print("\nEDA FOR CLEAN DATA\n")

from preprocessing_pipeline import df 
from utils import *

print("\nInfo About Nan's in Clean Data:\n")
print_nan_info(df)


numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop(["encounter_id", "patient_id", "hospital_id"])

num_cols = len(numerical_columns)
cols = 4  
rows = (num_cols + cols - 1) // cols 

plt.figure(figsize=(20, rows * 5)) 
for i, column in enumerate(numerical_columns):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df[column], color='skyblue')  
    plt.title(f'Boxplot of {column}')

plt.tight_layout() 
plt.savefig("assets/boxplot_clean_imputed_data.png")
 
 
plt.figure(figsize=(20, rows * 5))  
for i, column in enumerate(numerical_columns):
    plt.subplot(rows, cols, i + 1)
    sns.violinplot(x=df[column], color='skyblue', inner="quartile")  # Violin plot instead of boxplot
    plt.title(f'Violin Plot of {column}')

plt.tight_layout()  
plt.savefig("assets/violinplot_clean_imputed_data.png")

numeric_data = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', square=True, cbar_kws={'shrink': .75})
plt.title('Correlation Matrix Heatmap of Numerical Columns')
plt.savefig("assets/corr_matrix_clean_imputed_data.png")