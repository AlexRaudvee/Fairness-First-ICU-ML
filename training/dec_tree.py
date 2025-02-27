import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

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
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=69)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.savefig("assets/Dec_Tree_ConfMat.png")


print("Classification Report:\n", report)


# ROC AUC CURVE computation
# Get predicted probabilities for the positive class
y_prob = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig("assets/Dec_Tree_ROCAUC.png")


# DECISION TREE VIZ


plt.figure(figsize=(20, 10))
plot_tree(clf, max_depth=5, filled=True, feature_names=X_final.columns, 
          class_names=['dies', 'survives'], rounded=True)
plt.title('Decision Tree Visualization')
plt.savefig("assets/Dec_Tree_Viz.png")


# FEATURE IMPORTANCE BAR CHART


# Get feature importances from the classifier
features = X_final.columns
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align="center", color='teal')
plt.xticks(range(len(features)), features[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig("assets/Dec_Tree_FeatImp.png")
