import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


from DATA import data
# Separate features and target variable
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']                 # Target

models = []
models.append(('LR', LogisticRegression(random_state=12345)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=12345)))
models.append(('RF', RandomForestClassifier(random_state=12345)))
models.append(('SVM', SVC(gamma='auto', random_state=12345)))
models.append(('XGB', GradientBoostingClassifier(random_state=12345)))
models.append(("LightGBM", LGBMClassifier(random_state=12345)))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=12345)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
        
# Boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
