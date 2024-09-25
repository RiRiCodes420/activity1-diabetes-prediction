import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

# Load your dataset
data = pd.read_csv('your_file.csv')  # Adjust the filename as necessary

# Rename columns for clarity
data = data.rename(columns={'BMI': 'NewBMI', 'Insulin': 'NewInsulinScore', 'Glucose': 'NewGlucose'})

# One-hot encode categorical variables
df = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Define target and features
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Define models for evaluation with specified hyperparameters
models = []
models.append(('RF', RandomForestClassifier(random_state=12345, max_depth=8, max_features=7, min_samples_split=2, n_estimators=500)))
models.append(('XGB', GradientBoostingClassifier(random_state=12345, learning_rate=0.1, max_depth=5, min_samples_split=0.1, n_estimators=100, subsample=1.0)))
models.append(("LightGBM", LGBMClassifier(random_state=12345, learning_rate=0.01, max_depth=3, n_estimators=1000)))

# Evaluate each model
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=12345, shuffle=True)  # Shuffle for better randomness
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

