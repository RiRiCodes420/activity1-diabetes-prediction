import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# Load the data
data = pd.read_csv('your_file.csv')  # Replace with your actual data file
print("Initial columns in the data:", data.columns)

# Rename columns for consistency
data = data.rename(columns={'BMI': 'NewBMI', 'Insulin': 'NewInsulinScore', 'Glucose': 'NewGlucose'})

# Create dummy variables for categorical columns
# The drop_first=True will avoid the dummy variable trap
df = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Check the columns after one-hot encoding
print("Columns after one-hot encoding:", df.columns)

# Identify categorical columns created by get_dummies
categorical_columns = [col for col in df.columns if 'NewBMI_' in col or 'NewInsulinScore_' in col or 'NewGlucose_' in col]
print("Categorical columns created:", categorical_columns)

# Define target and features
y = df["Outcome"]
X = df.drop(["Outcome"] + categorical_columns, axis=1)

# Store columns and index for later use
cols = X.columns
index = X.index

# Standardize features using RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns=cols, index=index)

# Concatenate the categorical variables back to the feature set
X = pd.concat([X, df[categorical_columns]], axis=1)

# Check the target variable
print("First few rows of the target variable (y):")
print(y.head())

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
