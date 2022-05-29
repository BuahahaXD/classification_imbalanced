# %%
# # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
# %%
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]]#.values
y = dataset.iloc[:, -1]#.values

X1 = dataset.iloc[:, [2]]#age
X2 = dataset.iloc[:, [3]]#est_salary

colors = ['red','green']
plt.scatter(X1, X2, c=y, cmap=matplotlib.colors.ListedColormap(colors))

plt.show()

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# %%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
# SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
print(type(smote))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train = smote.fit_transform(X_train)

# %%
# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# %%
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print(cm)

print(ac)
# %%
