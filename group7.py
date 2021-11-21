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
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

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

X = np.vstack((income, beds, baths, area))
X = np.transpose(X)
X = np.hstack((X,type_enc))

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy()

# # Legend information for logistic regression model
# legend_markers = [
#     Line2D([0], [0], color='green', linewidth=0, marker='+'),
#     Line2D([0], [0], color='blue', linewidth=0, marker='o')
# ]
# legend_labels = ["+1", "-1"]

# # Plot data
# plt.rc('font', size=18)
# plt.rcParams['figure.constrained_layout.use'] = True
# for i in range(len(X)):
#     if y[i] == 1:
#         plt.scatter(X1[i], X2[i], color='green', marker='+')
#     else:
#         plt.scatter(X1[i], X2[i], color='blue', marker='o')
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend(legend_markers, legend_labels, loc=3)
# plt.show()

# # Cross validation - polynomial
# mean_error=[]; std_error=[]
# # Polynomial orders to validate
# PL_range = [1, 2, 3, 4, 5]
# # Loop for each value
# for Pi in PL_range:
#     poly = PolynomialFeatures(Pi)
#     Xp = poly.fit_transform(X)
#     model = linear_model.LogisticRegression(penalty='l2', C=15)
#     temp=[]
#     kf = KFold(n_splits=5)
#     # Divide data into train and test
#     for train, test in kf.split(Xp):
#         model.fit(Xp[train], y[train])
#         ypred = model.predict(Xp[test])
#         temp.append(f1_score(y[test], ypred, average='micro'))
#     # Calculate errors
#     mean_error.append(np.array(temp).mean())
#     std_error.append(np.array(temp).std())
# # Plot mean square error vs C value
# plt.errorbar(PL_range,mean_error,yerr=std_error)
# plt.xlabel('Pi'); plt.ylabel('F1 Score')
# plt.xlim((0,5))
# plt.title('L2 LR - Polynomial Cross Val')
# plt.show()

# C value cross validation
mean_error=[]; std_error=[]
# C values to validate
CL_range = [0.1, 0.5, 1, 5, 10, 15, 25, 50, 100]
# Loop for each value
for Ci in CL_range:
    model = linear_model.LogisticRegression(penalty='l2', C=Ci, max_iter=10000)
    temp=[]
    kf = KFold(n_splits=5)
    # Divide data into train and test
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(f1_score(y[test], ypred, average='micro'))
    # Calculate errors
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
# Plot mean square error vs C value
plt.errorbar(CL_range,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('F1 Score')
plt.xlim((0,100))
plt.title('L2 Regression - C Cross Validation')
plt.show()

# # Cross validation - KNN k
# mean_error=[]; std_error=[]
# # C values to validate
# K_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# # Loop for each k
# for Ki in K_range:
#     model = KNeighborsClassifier(n_neighbors=Ki, weights='uniform')
#     temp=[]
#     kf = KFold(n_splits=5)
#     # Divide data into train and test
#     for train, test in kf.split(X):
#         model.fit(X[train], y[train])
#         ypred = model.predict(X[test])
#         temp.append(f1_score(y[test], ypred, average='micro'))
#     # Calculate errors
#     mean_error.append(np.array(temp).mean())
#     std_error.append(np.array(temp).std())
# # Plot mean square error vs C value
# plt.errorbar(K_range,mean_error,yerr=std_error)
# plt.xlabel('Ki'); plt.ylabel('F1 Score')
# plt.xlim((0,10))
# plt.title('kNN - k Cross Validation')
# plt.show()

# # Legend information for logistic regression model
# legend_markers = [
#     Line2D([0], [0], color='green', linewidth=0, marker='+'),
#     Line2D([0], [0], color='blue', linewidth=0, marker='o'),
#     Line2D([0], [0], color='red', linewidth=0, marker='x'),
#     Line2D([0], [0], color='yellow', linewidth=0, marker='*')
# ]
# legend_labels = ["+1", "-1", "+1 Pred", "-1 Pred"]

# # Confusion matrix - logistic
# poly = PolynomialFeatures(2)
# Xp_train = poly.fit_transform(X_train)
# Xp_test = poly.fit_transform(X_test)
# # Train model and predict
# LRmodel = linear_model.LogisticRegression(penalty='l2', C=15)
# LRmodel.fit(Xp_train, y_train)
# y_predLR = LRmodel.predict(Xp_test)
# print("Intercept %f"%(LRmodel.intercept_))
# print("Coefficients ",(LRmodel.coef_))
# print('Logistic Regression Confusion Matrix')
# print(confusion_matrix(y_test, y_predLR))

# # Plot data and predictions
# plt.rc('font', size=18)
# plt.rcParams['figure.constrained_layout.use'] = True
# for i in range(len(X_train)):
#     if y_train[i] == 1:
#         plt.scatter(X_train[i, 0], X_train[i, 1], color='green', marker='+')
#     else:
#         plt.scatter(X_train[i, 0], X_train[i, 1], color='blue', marker='o')
# for i in range(len(X_test)):
#     if y_predLR[i] == 1:
#         plt.scatter(X_test[i, 0], X_test[i, 1], color='red', marker='x')
#     else:
#         plt.scatter(X_test[i, 0], X_test[i, 1], color='yellow', marker='*')
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend(legend_markers, legend_labels, loc=3)
# plt.show()

# # Confusion matrix - knn
# KNNmodel = KNeighborsClassifier(n_neighbors=5, weights='uniform')
# # Train model and predict
# KNNmodel.fit(X_train, y_train)
# y_predKNN = KNNmodel.predict(X_test)
# print('kNN Confusion Matrix')
# print(confusion_matrix(y_test, y_predKNN))

# # Plot data and predictions
# plt.rc('font', size=18)
# plt.rcParams['figure.constrained_layout.use'] = True
# for i in range(len(X_train)):
#     if y_train[i] == 1:
#         plt.scatter(X_train[i, 0], X_train[i, 1], color='green', marker='+')
#     else:
#         plt.scatter(X_train[i, 0], X_train[i, 1], color='blue', marker='o')
# for i in range(len(X_test)):
#     if y_predKNN[i] == 1:
#         plt.scatter(X_test[i, 0], X_test[i, 1], color='red', marker='x')
#     else:
#         plt.scatter(X_test[i, 0], X_test[i, 1], color='yellow', marker='*')
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend(legend_markers, legend_labels, loc=3)
# plt.show()

# # Baseline classifier
# BLmodel = DummyClassifier(strategy="uniform")
# BLmodel.fit(X_train, y_train)
# y_predBL = BLmodel.predict(X_test)
# print('Baseline Model Confusion Matrix')
# print(confusion_matrix(y_test, y_predBL))

# # Legend information for logistic regression model
# legend_markers = [
#     Line2D([0], [0], color='red', linewidth=1),
#     Line2D([0], [0], color='blue', linewidth=1),
#     Line2D([0], [0], color='green', linewidth=1, linestyle='--')
# ]
# legend_labels = ["Logistic Regression", "kNN", "Random Baseline"]

# # ROC Curve
# plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
# # Get curve for each model
# fprLR, tprLR, _ = roc_curve(y_test,LRmodel.decision_function(Xp_test))
# plt.plot(fprLR,tprLR,color='red')
# fprKNN, tprKNN, _ = roc_curve(y_test,KNNmodel.predict_proba(X_test)[:,1])
# plt.plot(fprKNN,tprKNN,color='blue')
# fprBL, tprBL, _ = roc_curve(y_test,BLmodel.predict_proba(X_test)[:,1])
# # Plot curve
# plt.plot(fprBL,tprBL, color='green',linestyle='--')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC Curves')
# plt.legend(legend_markers, legend_labels, loc=4)
# plt.show()