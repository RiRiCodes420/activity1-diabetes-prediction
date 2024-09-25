import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Diabetes1.csv')  # Ensure the path is correct

# Feature Engineering (As you already have)
# Assuming the feature engineering code is already present and the columns exist in 'data'

# Example of feature engineering (this part can be adjusted as per your previous steps)
# Define categories for BMI
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
data["NewBMI"] = NewBMI

# Categorize BMI
data.loc[data["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
data.loc[(data["BMI"] >= 18.5) & (data["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
data.loc[(data["BMI"] > 24.9) & (data["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
data.loc[(data["BMI"] > 29.9) & (data["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
data.loc[(data["BMI"] > 34.9) & (data["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
data.loc[data["BMI"] > 39.9, "NewBMI"] = NewBMI[5]

# Define insulin categories
def set_insulin(row):
    return "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal"

data["NewInsulinScore"] = data.apply(set_insulin, axis=1)

# Categorize glucose levels
data["NewGlucose"] = pd.cut(data["Glucose"],
                             bins=[-float('inf'), 70, 99, 126, float('inf')],
                             labels=["Low", "Normal", "Overweight", "High"],
                             right=True)
data["NewGlucose"] = data["NewGlucose"].cat.add_categories("Secret")
data.loc[data["Glucose"] > 126, "NewGlucose"] = "Secret"

# One-hot encoding of categorical features
data = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Prepare features and target variable
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=12345)
model = LogisticRegression()

accuracies = []

for train_index, test_index in kfold.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

# Output the results
print(f'Accuracies for each fold: {accuracies}')
print(f'Mean Accuracy: {sum(accuracies) / len(accuracies):.2f}')
