from DATA import data
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# Initialize the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=10)

# Fit the model and predict outliers
lof.fit_predict(data)

# Get the negative outlier factor scores
data_scores = lof.negative_outlier_factor_

# Sort the scores and display the first 30
print(np.sort(data_scores)[0:30])

# Define the threshold for outliers
threshold = np.sort(data_scores)[7]
print("Threshold:", threshold)

# Identify outliers
outlier = data_scores > threshold

# Filter the dataset to keep only inliers
data = data[outlier]

# Examine the size of the dataset
print("Shape of the dataset:", data.shape)
