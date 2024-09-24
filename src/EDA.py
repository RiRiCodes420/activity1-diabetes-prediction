import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import warnings
<<<<<<< HEAD
warnings.simplefilter(action="ignore")



# Display the first few rows of the dataset
data.head()

# Shape of the dataset
print(data.shape)

# Information about the dataset
data.info()

# Descriptive statistics
data.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# Value counts of the 'Outcome' column as a percentage
print(data["Outcome"].value_counts() * 100 / len(data))

# Value counts of the 'Outcome' column
print(data.Outcome.value_counts())

# Histogram of the 'Age' column
data["Age"].hist(edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Maximum and Minimum Age
print("Max Age: " + str(data["Age"].max()) + " Min Age: " + str(data["Age"].min()))

# Pie chart and count plot for 'Outcome'
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Target Distribution')
ax[0].set_ylabel('')
sns.countplot(x='Outcome', data=data, ax=ax[1])
ax[1].set_title('Count of Outcomes')
plt.show()

# Heatmap for correlation matrix
f, ax = plt.subplots(figsize=[20, 15])
sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
=======
warnings.simplefilter(action = "ignore") 

data.info()
>>>>>>> f817f8d085280456183f9e1145f2813a95c2c150
