import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

from DATA import data
# Define features and target variable
X = data.drop(columns=['Outcome'])  # Replace 'Outcome' with your actual target variable
y = data['Outcome']  # Replace 'Outcome' with your actual target variable

# Random Forest Classifier Tuning
rf_params = {
    "n_estimators": [100, 200, 500, 1000],
    "max_features": [3, 5, 7],
    "min_samples_split": [2, 5, 10, 30],
    "max_depth": [3, 5, 8, None]
}

rf_model = RandomForestClassifier(random_state=12345)
gs_cv_rf = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2)
gs_cv_rf.fit(X, y)

# Best parameters for Random Forest
print("Best parameters for Random Forest:", gs_cv_rf.best_params_)

# Train the Random Forest model with best parameters
rf_tuned = RandomForestClassifier(**gs_cv_rf.best_params_).fit(X, y)
rf_cv_score = cross_val_score(rf_tuned, X, y, cv=10).mean()
print(f"Random Forest Mean Cross-Validation Score: {rf_cv_score:.4f}")

# Feature importance for Random Forest
feature_imp_rf = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp_rf, y=feature_imp_rf.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Random Forest Variable Importance")
plt.show()

# Gradient Boosting Classifier Tuning
xgb = GradientBoostingClassifier(random_state=12345)
xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2, 1],
    "min_samples_split": np.linspace(0.1, 0.5, 10),
    "max_depth": [3, 5, 8],
    "subsample": [0.5, 0.9, 1.0],
    "n_estimators": [100, 1000]
}

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2)
xgb_cv_model.fit(X, y)

# Best parameters for Gradient Boosting
print("Best parameters for Gradient Boosting:", xgb_cv_model.best_params_)

# Train the Gradient Boosting model with best parameters
xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X, y)
xgb_cv_score = cross_val_score(xgb_tuned, X, y, cv=10).mean()
print(f"Gradient Boosting Mean Cross-Validation Score: {xgb_cv_score:.4f}")

# Feature importance for Gradient Boosting
feature_imp_xgb = pd.Series(xgb_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp_xgb, y=feature_imp_xgb.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Gradient Boosting Variable Importance")
plt.show()
