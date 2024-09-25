import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler

# Load your dataset
data = pd.read_csv('your_file.csv')  # Adjust the filename as necessary

# Rename columns for clarity
data = data.rename(columns={'BMI': 'NewBMI', 'Insulin': 'NewInsulinScore', 'Glucose': 'NewGlucose'})

# One-hot encode categorical variables
df = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Define target and features
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Standardize features using RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns=df.columns[1:], index=df.index)  # Adjust based on your columns

# Define models for evaluation
models = [
    ('LR', LogisticRegression(random_state=12345)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=12345)),
    ('RF', RandomForestClassifier(random_state=12345)),
    ('SVM', SVC(gamma='auto', random_state=12345)),
    ('XGB', GradientBoostingClassifier(random_state=12345)),
    ('LightGBM', LGBMClassifier(random_state=12345))
]

# Evaluate each model
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=12345, shuffle=True)  # Ensure shuffling for better randomness
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})"
    print(msg)

# Boxplot for algorithm comparison
plt.figure(figsize=(15, 10))
plt.suptitle('Algorithm Comparison')
plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names)
plt.ylabel('Accuracy')
plt.show()

# Hyperparameter tuning for Random Forest
rf_params = {
    "n_estimators": [100, 200, 500, 1000],
    "max_features": [3, 5, 7],
    "min_samples_split": [2, 5, 10, 30],
    "max_depth": [3, 5, 8, None]
}

rf_model = RandomForestClassifier(random_state=12345)

gs_cv = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2)
gs_cv.fit(X, y)
print("Best parameters for Random Forest:", gs_cv.best_params_)

rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
print("Random Forest cross-validated score:", cross_val_score(rf_tuned, X, y, cv=10).mean())

# Feature importance for Random Forest
feature_imp = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Random Forest Variable Severity Levels")
plt.show()

# Hyperparameter tuning for LGBM
lgbm = LGBMClassifier(random_state=12345)
lgbm_params = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
    "n_estimators": [500, 1000, 1500],
    "max_depth": [3, 5, 8]
}

gs_cv = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=2)
gs_cv.fit(X, y)
print("Best parameters for LGBM:", gs_cv.best_params_)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
print("LGBM cross-validated score:", cross_val_score(lgbm_tuned, X, y, cv=10).mean())

# Feature importance for LGBM
feature_imp = pd.Series(lgbm_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("LGBM Variable Severity Levels")
plt.show()

# Hyperparameter tuning for Gradient Boosting
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
print("Best parameters for Gradient Boosting:", xgb_cv_model.best_params_)

xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X, y)
print("Gradient Boosting cross-validated score:", cross_val_score(xgb_tuned, X, y, cv=10).mean())

# Feature importance for Gradient Boosting
feature_imp = pd.Series(xgb_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Gradient Boosting Variable Severity Levels")
plt.show()


