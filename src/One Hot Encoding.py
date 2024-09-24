import pandas as pd
from DATA import data

# Define the categories for BMI and assign them to the 'NewBMI' column
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
data["NewBMI"] = NewBMI

data.loc[data["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
data.loc[(data["BMI"] >= 18.5) & (data["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
data.loc[(data["BMI"] > 24.9) & (data["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
data.loc[(data["BMI"] > 29.9) & (data["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
data.loc[(data["BMI"] > 34.9) & (data["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
data.loc[data["BMI"] > 39.9, "NewBMI"] = NewBMI[5]

# Display the first few rows to check the new column
print("Data with NewBMI:")
print(data.head())

# Define a function to categorize insulin levels
def set_insulin(row):
    if 16 <= row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"

# Assign the insulin score using the correct DataFrame reference
data["NewInsulinScore"] = data.apply(set_insulin, axis=1)

# Display the first few rows to check the new column
print("Data with NewInsulinScore:")
print(data.head())

# Define the categories for glucose levels and assign them to the 'NewGlucose' column
data["NewGlucose"] = pd.cut(data["Glucose"],
                            bins=[-float('inf'), 70, 99, 126, float('inf')],
                            labels=["Low", "Normal", "Overweight", "High"],
                            right=True)

# Convert NewGlucose to categorical with the "Secret" category
data["NewGlucose"] = data["NewGlucose"].cat.add_categories("Secret")

# Introduce the custom "Secret" category for Glucose > 126
data.loc[data["Glucose"] > 126, "NewGlucose"] = "Secret"

# Display the first few rows to check the new column
print("Data with NewGlucose:")
print(data.head())

# One-hot encode the new categorical columns
data_encoded = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Display the encoded DataFrame
print("Data after one-hot encoding:")
print(data_encoded.head())

# Optionally, display all columns in the encoded DataFrame
print("Columns in the encoded DataFrame:", data_encoded.columns)
