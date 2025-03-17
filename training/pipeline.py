import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve, f1_score,
                             average_precision_score, brier_score_loss, roc_auc_score)
from sklearn.calibration import calibration_curve
from utils import *
import shap
from lime.lime_tabular import LimeTabularExplainer


class CustomPipeline:
    def __init__(self, model_type: str = "logreg"):
        self.model_type = model_type
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_prob = None
        self.df = None
        self.reweighting = False
        
        if self.model_type == "logreg":
            self.model = LogisticRegression(random_state=69, class_weight="balanced", max_iter=10000)
        elif self.model_type == "dectree":
            self.model = DecisionTreeClassifier(criterion='gini', random_state=42)
        else:
            raise TypeError(f"{self.model_type} is not yet supported, check the docs")

        self.sensitive_features = ['age', 'bmi', 'ethnicity', 'gender', 'height',
                              'apache_2_diagnosis', 'aids', 'cirrhosis',
                              'diabetes_mellitus', 'immunosuppression', 'leukemia',
                              'lymphoma', 'solid_tumor_with_metastasis']

    def preprocessing(self, data_path: str, desc_path: str, target: str):
        
        # PREPROCESSING PIPELINE

        self.df: pd.DataFrame = pd.read_csv(data_path)
        df_: pd.DataFrame = pd.read_csv(desc_path)

        BINARY_ATTS = df_[df_['Data Type'].str.lower() == 'binary']['Variable Name'].tolist()
        print(f"Binary atts : {BINARY_ATTS}")
        STRING_ATTS = df_[df_['Data Type'].str.lower() == 'string']['Variable Name'].tolist()
        print(f"String atts : {STRING_ATTS}")
        INT_ATTS = df_[df_['Data Type'].str.lower() == 'integer']['Variable Name'].tolist()
        print(f"Integer atts : {INT_ATTS}")
        FLOAT_ATTS = df_[df_['Data Type'].str.lower() == 'numeric']['Variable Name'].tolist()
        FLOAT_ATTS.remove('pred')
        print(f"Float atts : {FLOAT_ATTS}")

        # drop the id columns and those that are non relevant for the predictions (noise or baseline predictions)
        self.df = self.df.drop(columns=["encounter_id", "patient_id", "hospital_id", "icu_id", "icu_stay_type", "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"])

        print("\nDrop features with high NANs: ")
        print(f"Feature Number Before: {len(self.df.columns)}")
        self.df, columns_to_drop = drop_high_NaN_features(self.df)
        print(f"Feature Number After: {len(self.df.columns)}")

        print("\nImpute Remaining Missing Values: ")
        print(f"Total NANs Before: \n{self.df.isnull().sum().sort_values(ascending=False)}")
        self.df = impute_values_for_features(self.df)
        print(f"Total NANs After: \n{self.df.isnull().sum().sort_values(ascending=False)}")

        # print("\nFilter out outliers: \n")
        # print(f"Shape of the Dataframe before: {self.df.shape}")
        # self.df = drop_outliers(self.df, int_cols=INT_ATTS, float_cols=FLOAT_ATTS)
        # print(f"Shape of the Dataframe after: {self.df.shape}")

        print("\nScale The Data For Better Visualization: ")
        print(f"Description Before: \n{self.df.select_dtypes(include=['int64', 'float64']).describe(include='all')}")
        self.df = standard_scaling_of_features(self.df, int_cols=INT_ATTS, float_cols=FLOAT_ATTS)
        print(f"Description After: \n{self.df.select_dtypes(include=['int64', 'float64']).describe(include='all')}")

        print("\nDrop Highly Correlated Features: ")
        print(f"Feature Number Before: {len(self.df.columns)}")
        self.df, to_drop = drop_highly_correlated_features(self.df)
        print(f"Feature Number After: {len(self.df.columns)}")
        print(to_drop)
        
        print("\n Save the DF: ")
        save_path = get_path_until_data(data_path) + "cleaned_imputed.csv"
        self.df.to_csv(save_path)
        print(f"Saved at: {save_path}")
        
        # Filter out only those columns that exist in the DataFrame
        int_cols = [col for col in BINARY_ATTS+INT_ATTS if col in self.df.columns]
        str_cols = [col for col in STRING_ATTS if col in self.df.columns]
        float_cols = [col for col in FLOAT_ATTS if col in self.df.columns]

        # Convert the existing columns to numeric and float for memory optimization, coercing errors to NaN 
        self.df[int_cols] = self.df[int_cols].apply(lambda col: pd.to_numeric(col, errors='raise', downcast="integer"))
        self.df[float_cols] = self.df[float_cols].apply(lambda col: pd.to_numeric(col, errors='raise', downcast="float"))
        
        print(self.df.columns)
        
        # Separate features and target
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # Apply Reweighing
        self.sample_weights = self.apply_reweighing(X, y)


        # Identify categorical and numerical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # One-Hot Encode categorical variables
        self.ohe = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' avoids dummy variable trap
        X_encoded = pd.DataFrame(self.ohe.fit_transform(X[cat_cols]))
        
        # Restore column names after encoding
        X_encoded.columns = self.ohe.get_feature_names_out(cat_cols)
        print(X_encoded.columns)
        # Combine numerical and encoded categorical features
        self.X_final = pd.concat([X_encoded, X[num_cols].reset_index(drop=True)], axis=1)

        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_final, y, test_size=0.2, random_state=69)

        print(f"""
              X train: {self.X_train.shape}\n
              X test: {self.X_test.shape}\n
              y train: {self.y_train.shape}\n
              y test: {self.y_test.shape}\n
              """)

    def apply_reweighing(self, X, y):
        """
        Compute reweighting factors based on multiple sensitive attributes,
        handling both numerical and categorical sensitive features.
        """
        weights = np.ones(len(y))

        for feature in self.sensitive_features:
            if X[feature].dtype in [np.float64, np.int64]:
                privileged = X[feature].median()
                p_obs = y[X[feature] >= privileged].mean()
            else:
                most_frequent = X[feature].mode()[0]
                p_obs = y[X[feature] == most_frequent].mean()

            p_exp = y.mean()
            factor = p_exp / (p_obs + 1e-6)  # Avoid division by zero

            if X[feature].dtype in [np.float64, np.int64]:
                weights[X[feature] >= privileged] *= factor
            else:
                weights[X[feature] == most_frequent] *= factor

        return pd.Series(weights, index=X.index)


    def nested_cross_validation(self):
        """
        Apply nested cross-validation with reweighing.
        """
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # param_grid = {
        #     'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        #     'l1_ratio': [0.0, 0.5, 1.0],
        #     # 'C': [0.1, 1, 10],
        #     'solver': ['liblinear', 'saga', 'newton-cg'],
        #     'class_weight': [None, {0:1, 1:3}, {0:1, 1:5}, 'balanced', 'reweighting'],
        #     # 'class_weight': [None, 'balanced', 'reweighting'],
        # }

        param_grid = [
            {'penalty': ['l1', 'l2'], 'solver': ['liblinear'],
             'C' : [0.1, 1, 10],'class_weight': [None, {0: 1, 1: 3}, {0: 1, 1: 5}, 'balanced']},
            {'penalty': ['l2' ], 'solver': ['newton-cg'],
             'C': [0.1, 1, 10],'class_weight': [None, {0: 1, 1: 3}, {0: 1, 1: 5}, 'balanced']},
            {'penalty': ['l1', 'l2'], 'solver': ['saga'], 'C' : [0.1, 1, 10],
             'class_weight': [None, {0: 1, 1: 3}, {0: 1, 1: 5}, 'balanced']},
            {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.0, 0.5, 1.0], 'C' : [0.1, 1, 10],
             'class_weight': [None, {0: 1, 1: 3}, {0: 1, 1: 5}, 'balanced']}
        ]

        best_models = []
        best_scores = []

        for train_idx, test_idx in outer_cv.split(self.X_train, self.y_train):
            X_train, X_test = self.X_train.iloc[train_idx], self.X_train.iloc[test_idx]
            y_train, y_test = self.y_train.iloc[train_idx], self.y_train.iloc[test_idx]

            model = LogisticRegression(random_state=69, max_iter=10000)
            grid_search = GridSearchCV(model,param_grid,
                                       scoring={'recall': 'recall', 'f1': 'f1'}, refit='f1',
                                       cv=inner_cv, verbose = 2, error_score='raise', n_jobs=3)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_models.append(best_model)

            y_pred = best_model.predict(X_test)
            score = f1_score(y_test, y_pred)
            best_scores.append(score)

        best_idx = np.argmax(best_scores)
        self.model = best_models[best_idx]
        print(f"Best model selected with AU-ROC: {best_scores[best_idx]:.4f}")

        if grid_search.best_params_["class_weight"] == "reweighting":
            self.reweighting = True
        elif grid_search.best_params_["class_weight"] == "balanced":
            self.reweighting = False
        else:
            self.reweighting = False

        with open("nested_cv_best_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        print("Best nested CV model saved as nested_cv_best_model.pkl")
        with open("nested_cv_hyperparams.txt", "w") as f:
            f.write(str(grid_search.best_params_))
        print("Best hyperparameters saved in nested_cv_hyperparams.txt")

    def train(self, apply_reweighting=False):
        if apply_reweighting:
            self.X_train = self.ohe.inverse_transform(self.X_train)
            self.y_train = self.ohe.inverse_transform(self.y_train)
            self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        else:
            self.model.fit(self.X_train, self.y_train)
        print("Training is Done")

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        if self.model_type == "logreg":
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
    def eval(self):

        # Calculate evaluation metrics
        if self.model_type == "logreg":
            accuracy = accuracy_score(self.y_test, self.y_pred)
            conf_matrix = confusion_matrix(self.y_test, self.y_pred)
            report = classification_report(self.y_test, self.y_pred)

            print(f"""
                Accuracy: {accuracy}\n
                Confusion Matrix: \n{conf_matrix}\n
                Report: \n{report}\n
                """)
            
            # AUC-ROC
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
            roc_auc = auc(fpr, tpr)
            print(f"ROC AUC: {roc_auc:.3f}")

            # Plot ROC
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig("../assets/Log_Reg_ROCAUC.png")
            
            # F1 Score & Precision-Recall
            f1 = f1_score(self.y_test, self.y_pred)
            precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
            avg_precision = average_precision_score(self.y_test, self.y_prob)
            print(f"F1 Score: {f1:.3f}")
            print(f"Average Precision (PR AUC): {avg_precision:.3f}")

            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig("../assets/Log_Reg_PrecRecCur.png")
            
            # Calibration (Brier Score + Calibration Plot)
            brier = brier_score_loss(self.y_test, self.y_prob)
            print(f"Brier Score: {brier:.3f} (lower = better calibrated)")

            prob_true, prob_pred = calibration_curve(self.y_test, self.y_prob, n_bins=10)
            plt.figure(figsize=(6, 5))
            plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot')
            plt.legend()
            plt.savefig("../assets/Log_Reg_CalScore.png")
            
            # Reconstruct test set from original df using the SAME index
            df_test = self.df.loc[self.X_test.index]  # <--- DO NOT reset the index

            # Convert y_pred to a Series aligned with X_test.index
            y_pred_test = pd.Series(self.y_pred, index=self.X_test.index)

            # Suppose 'gender' is in df and we define privileged = 'M', unprivileged = 'F'
            priv_mask = (df_test['gender'] == 'M')
            unpriv_mask = (df_test['gender'] == 'F')

            # Demographic Parity
            priv_selection_rate = y_pred_test[priv_mask].mean()
            unpriv_selection_rate = y_pred_test[unpriv_mask].mean()
            dp_diff = priv_selection_rate - unpriv_selection_rate

            print(f"Demographic Parity difference = {dp_diff:.3f}")
            print(f"Privileged group selection rate = {priv_selection_rate:.3f}")
            print(f"Unprivileged group selection rate = {unpriv_selection_rate:.3f}")

            # Equalized Odds
            def compute_tpr_fpr(y_true_group, y_pred_group):
                cm = confusion_matrix(y_true_group, y_pred_group)
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0
                return tpr, fpr

            tpr_priv, fpr_priv = compute_tpr_fpr(self.y_test[priv_mask], y_pred_test[priv_mask])
            tpr_unpriv, fpr_unpriv = compute_tpr_fpr(self.y_test[unpriv_mask], y_pred_test[unpriv_mask])

            print(f"Equalized Odds TPR difference = {tpr_priv - tpr_unpriv:.3f}")
            print(f"TPR (priv) = {tpr_priv:.3f}, TPR (unpriv) = {tpr_unpriv:.3f}")
            print(f"Equalized Odds FPR difference = {fpr_priv - fpr_unpriv:.3f}")
            print(f"FPR (priv) = {fpr_priv:.3f}, FPR (unpriv) = {fpr_unpriv:.3f}")

            # Logistic Regression Coefficients Bar Chart
            # Visualize the weight of each feature, which helps in model interpretability.
            coefficients = self.model.coef_[0]
            features = self.X_final.columns

            # Create a DataFrame for better visualization
            coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
            coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
            plt.title('Logistic Regression Feature Coefficients')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.savefig("../assets/Log_Reg_FeatImp.png")
            
        elif self.model_type == "dectree": 
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(self.y_test, self.y_pred)
            conf_matrix = confusion_matrix(self.y_test, self.y_pred)
            report = classification_report(self.y_test, self.y_pred)

            # Print the evaluation results
            print(f"""
                  Accuracy: {accuracy}\n 
                  Report: \n {report}\n
                  """)
            # Compute confusion matrix
            conf_matrix = confusion_matrix(self.y_test, self.y_pred)

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('Actual Label')
            plt.title('Confusion Matrix')
            plt.savefig("../assets/Dec_Tree_ConfMat.png")

            # ROC AUC CURVE computation
            # Get predicted probabilities for the positive class
            y_prob = self.model.predict_proba(self.X_test)[:, 1]

            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.savefig("../assets/Dec_Tree_ROCAUC.png")


            # DECISION TREE VIZ
            plt.figure(figsize=(20, 10))
            plot_tree(self.model, max_depth=5, filled=True, feature_names=self.X_final.columns, 
                    class_names=['dies', 'survives'], rounded=True)
            plt.title('Decision Tree Visualization')
            plt.savefig("../assets/Dec_Tree_Viz.png")


            # FEATURE IMPORTANCE BAR CHART
            # Get feature importances from the classifier
            features = self.X_final.columns
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 12))
            plt.title("Feature Importances")
            plt.bar(range(len(features)), importances[indices], align="center", color='teal')
            plt.xticks(range(len(features)), features[indices], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.savefig("../assets/Dec_Tree_FeatImp.png")
        
        else: 
            raise ValueError("Ooops... Something went wrong, check the model_type during initialization")
        
    def explain_model(self):
        try:
            # SHAP
            explainer = shap.Explainer(self.model, self.X_train)
            shap_values = explainer(self.X_test)

            # Plot SHAP 
            shap.summary_plot(shap_values, self.X_test, plot_type="bar")
            plt.title('SHAP Summary Plot')
            plt.savefig('../assets/shap_summary_plot.png')
            plt.clf()

            # LIME 
            explainer = LimeTabularExplainer(self.X_train.values,
                                         mode='classification',
                                         training_labels=self.y_train,
                                         feature_names=self.X_train.columns.tolist(),
                                         class_names=['Negative', 'Positive'],
                                         random_state=42)

            # Explain a random instance
            i = np.random.randint(0, self.X_test.shape[0])
            exp = explainer.explain_instance(self.X_test.iloc[i].values, self.model.predict_proba, num_features=10)
            exp.as_pyplot_figure()
            plt.title(f'LIME Explanation {i}')
            plt.savefig(f'../assets/lime_explanation_instance_{i}.png')

            print(f"LIME explanation for instance {i} saved.")
            return self
    
        except Exception as e:
            print("Failed to generate or save plot:", e)

        
    
    def __str__(self):
        return f"Model: {self.model_type}, data: {self.df}"
    
    def __repr__(self):
        return f"Model: {self.model_type}, data: {self.df}"
    
pipe = CustomPipeline("logreg")
pipe.preprocessing("../physionet.org/files/widsdatathon2020/1.0.0/data/training_v2.csv", "../physionet.org/files/widsdatathon2020/1.0.0/data/WiDS_Datathon_2020_Dictionary.csv", "hospital_death")
pipe.nested_cross_validation()


print("Model WITHOUT reweighting:")
pipe.train(apply_reweighting=False)
pipe.predict()
pipe.eval()

print("Model WITH reweighting:")
pipe.train(apply_reweighting=True)
pipe.predict()
pipe.eval()

pipe.explain_model()
