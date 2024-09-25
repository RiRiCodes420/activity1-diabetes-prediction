import gdown
import pandas as pd

# Google Drive file ID
file_url = 'https://drive.google.com/uc?id=1mv1_iw0HqngqLUxqCnhy11AnNTCNcSKH'

# Download the file using gdown
output = 'your_file.csv'  # You can specify any file name here
gdown.download(file_url, output, quiet=False)

# Now read the CSV file into a pandas DataFrame
data = pd.read_csv(output)

# Display the first few rows of the data to confirm
print(data.head())

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
warnings.simplefilter(action = "ignore") 

print(data.shape)
print(data.info())
print(data.describe())

data["Age"].hist(edgecolor = "black");
plt.show()

print("Max Age: " + str(data["Age"].max()) + " Min Age: " + str(data["Age"].min()))
import seaborn as sns
import matplotlib.pyplot as plt

# Create a grid of subplots (4 rows, 2 columns)
fig, ax = plt.subplots(4, 2, figsize=(16,16))

# Plot each feature's distribution on a separate subplot
sns.histplot(data.Age, bins=20, kde=True, ax=ax[0,0]) 
sns.histplot(data.Pregnancies, bins=20, kde=True, ax=ax[0,1]) 
sns.histplot(data.Glucose, bins=20, kde=True, ax=ax[1,0]) 
sns.histplot(data.BloodPressure, bins=20, kde=True, ax=ax[1,1]) 
sns.histplot(data.SkinThickness, bins=20, kde=True, ax=ax[2,0])
sns.histplot(data.Insulin, bins=20, kde=True, ax=ax[2,1])
sns.histplot(data.DiabetesPedigreeFunction, bins=20, kde=True, ax=ax[3,0]) 
sns.histplot(data.BMI, bins=20, kde=True, ax=ax[3,1])
plt.tight_layout()
plt.show()

data.groupby("Outcome").agg({"Pregnancies":"mean"})

data.groupby("Outcome").agg({"Age":"mean"})

data.groupby("Outcome").agg({"Age":"max"})

data.groupby("Outcome").agg({"Insulin": "mean"})

data.groupby("Outcome").agg({"Insulin": "max"})

data.groupby("Outcome").agg({"Glucose": "mean"})

data.groupby("Outcome").agg({"Glucose": "max"})

data.groupby("Outcome").agg({"BMI": "mean"})

f,ax=plt.subplots(1,2,figsize=(18,8))
data['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('Outcome', ax=ax[1])
ax[1].set_title('Outcome')
plt.show()

f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()