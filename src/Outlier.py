from DATA import data
for feature in data:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    if (data[feature] > upper).any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")

import seaborn as sns
sns.boxplot(x = data["Insulin"]);

Q1 = data.Insulin.quantile(0.25)
Q3 = data.Insulin.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Replace outliers in the "Insulin" column with the upper limit
data.loc[data["Insulin"] > upper, "Insulin"] = upper

import seaborn as sns
sns.boxplot(x = data["Insulin"]);