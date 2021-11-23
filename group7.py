## Group 7 Machine Learning Project
## David Dunne - 
## Eoin Lynch - 
## Charlie Maguire - 17332641

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, mean_squared_error

df = pd.read_csv("ml-dataset.csv")
print(df.head())
y = df.iloc[:,1]
income = df.iloc[:,2]
beds = df.iloc[:,3]
baths = df.iloc[:,4]
area = df.iloc[:,5]
p_type = df.iloc[:,6]

# Peform one-hot encoding on property types
lab_enc = LabelEncoder()
type_int = lab_enc.fit_transform(p_type)
type_int = type_int.reshape(len(type_int), 1)
# Uncomment for house types
# print(list(lab_enc.classes_))
# print(type_int)
oh_enc = OneHotEncoder(sparse=False)
type_enc = oh_enc.fit_transform(type_int)

scaler = StandardScaler()
income = (income-income.min())/(income.max()-income.min())
area = (area-area.min())/(area.max()-area.min())
beds = (beds-beds.min())/(beds.max()-beds.min())
baths = (baths-baths.min())/(baths.max()-beds.min())

X = np.vstack((income, beds, baths, area))
X = np.transpose(X)
X = scaler.fit_transform(X)
print(X)

# X = np.hstack((X,type_enc))

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# C value cross validation
mean_error=[]; std_error=[]
# C values to validate
CL_range = [0.0001,0.001,0.01,0.1, 0.5, 1]# Loop for each value
for Ci in CL_range:
    model = linear_model.LinearRegression()
    temp=[]
    kf = KFold(n_splits=5)
    # Divide data into train and test
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],ypred))    # Calculate errors
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
# Plot mean square error vs C value
plt.errorbar(CL_range,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('MSE')
plt.xlim((0,1))
plt.title('Linear Regression - C Cross Validation')
plt.show()

# C value cross validation
mean_error=[]; std_error=[]
# C values to validate
CL_range = [0.0001,0.001,0.01,0.1, 0.5, 1]# Loop for each value
for Ci in CL_range:
    model = linear_model.Lasso(alpha=1/(2*Ci), max_iter=1000)
    temp=[]
    kf = KFold(n_splits=5)
    # Divide data into train and test
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],ypred))    # Calculate errors
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
# Plot mean square error vs C value
plt.errorbar(CL_range,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('MSE')
plt.xlim((0,1))
plt.title('Lasso Regression - C Cross Validation')
plt.show()

# C value cross validation
mean_error=[]; std_error=[]
# C values to validate
CL_range = [0.0001,0.001,0.01,0.1, 0.5, 1]# Loop for each value
for Ci in CL_range:
    model = linear_model.Ridge(alpha=1/(2*Ci), max_iter=1000)
    temp=[]
    kf = KFold(n_splits=5)
    # Divide data into train and test
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],ypred))
    # Calculate errors
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
# Plot mean square error vs C value
plt.errorbar(CL_range,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('MSE')
plt.xlim((0,1))
plt.title('Ridge Regression - C Cross Validation')
plt.show()

# Cross validation - KNN k
mean_error=[]; std_error=[]
# C values to validate
K_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Loop for each k
for Ki in K_range:
    model = KNeighborsClassifier(n_neighbors=Ki, weights='uniform')
    temp=[]
    kf = KFold(n_splits=5)
    # Divide data into train and test
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],ypred))
    # Calculate errors
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
# Plot mean square error vs C value
plt.errorbar(K_range,mean_error,yerr=std_error)
plt.xlabel('Ki'); plt.ylabel('MSE')
plt.xlim((0,10))
plt.title('kNN - k Cross Validation')
plt.show()
