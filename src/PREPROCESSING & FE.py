import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
import missingno as msno

# Load the data
data = pd.read_csv('your_file.csv')  

# Replace zeros with NaN for certain columns
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

# Check and visualize missing values
print("Missing values before filling:")
print(data.isnull().sum())
msno.bar(data)

# Function to fill missing values with median based on Outcome
def fill_missing_with_median(data, var):
    median_values = data.groupby('Outcome')[var].median()
    data.loc[(data['Outcome'] == 0) & (data[var].isnull()), var] = median_values[0]
    data.loc[(data['Outcome'] == 1) & (data[var].isnull()), var] = median_values[1]

# Fill missing values for specified columns
columns = data.columns.drop("Outcome")
for col in columns:
    fill_missing_with_median(data, col)

# Check for missing values again
print("Missing values after filling:")
print(data.isnull().sum())

# Outlier detection using IQR
# Outlier detection using IQR
for feature in data:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Check for outliers
    outliers_detected = (data[feature] > upper)
    if outliers_detected.any():
        print(f"{feature}: Outliers detected")
    else:
        print(f"{feature}: No outliers detected")


# Visualize Insulin distribution before capping outliers
import matplotlib.pyplot as plt  # Add this line

# ... your existing code ...

# Visualize Insulin distribution before capping outliers
sns.boxplot(x=data["Insulin"])
plt.show()  # Use plt.show() to display the plot

# Cap outliers in Insulin
data.loc[data["Insulin"] > upper, "Insulin"] = upper

# Visualize Insulin distribution after capping
sns.boxplot(x=data["Insulin"])
plt.show()  # Use plt.show() to display the plot

# Cap outliers in Insulin
data.loc[data["Insulin"] > upper, "Insulin"] = upper

# Visualize Insulin distribution after capping
sns.boxplot(x=data["Insulin"])
plt.show()  # Ensure the plot displays if running in an interactive environment

# Local Outlier Factor for additional outlier detection
lof = LocalOutlierFactor(n_neighbors=10)
data_scores = lof.fit_predict(data)

# Sorting and determining threshold for outliers
threshold = np.sort(lof.negative_outlier_factor_)[7]
print("Threshold for outliers:", threshold)

# Keep only non-outliers
data = data[data_scores > threshold]

# Create BMI categories
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
data["NewBMI"] = NewBMI

data.loc[data["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
data.loc[(data["BMI"] >= 18.5) & (data["BMI"] < 24.9), "NewBMI"] = NewBMI[1]
data.loc[(data["BMI"] >= 24.9) & (data["BMI"] < 29.9), "NewBMI"] = NewBMI[2]
data.loc[(data["BMI"] >= 29.9) & (data["BMI"] < 34.9), "NewBMI"] = NewBMI[3]
data.loc[(data["BMI"] >= 34.9) & (data["BMI"] < 39.9), "NewBMI"] = NewBMI[4]
data.loc[data["BMI"] >= 39.9, "NewBMI"] = NewBMI[5]

# Create insulin category
def set_insulin(row):
    return "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal"

data["NewInsulinScore"] = data.apply(set_insulin, axis=1)

# Create glucose categories
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype="category")
data["NewGlucose"] = NewGlucose

data.loc[data["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
data.loc[(data["Glucose"] > 70) & (data["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
data.loc[(data["Glucose"] > 99) & (data["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
data.loc[data["Glucose"] > 126, "NewGlucose"] = NewGlucose[3]

# Display the first few rows of the modified DataFrame
print(data.head())
