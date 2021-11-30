# Group 7 Machine Learning Project
# David Dunne - 17329756
# Eoin Lynch -
# Charlie Maguire - 17332641

# Imports
import numpy as np
from numpy.lib.utils import deprecate
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, mean_squared_error, r2_score

LASSO_C_VALUE = 0.01
RIDGE_C_VALUE = 10
K_VALUE = 4 # 7 also good

def main(dataset):
    df = pd.read_csv(dataset)
    print(df.head())
    y = df.iloc[:, 1]
    income = df.iloc[:, 2]
    beds = df.iloc[:, 3]
    baths = df.iloc[:, 4]
    area = df.iloc[:, 5]
    p_type = df.iloc[:, 6]

    # Good for Lasso Regression Convergence, don't think normalisation needed for others
    income = normalise(income)
    beds = normalise(beds)
    baths = normalise(baths)
    area = normalise(area)

    # Peform one-hot encoding on property types
    lab_enc = LabelEncoder()
    type_int = lab_enc.fit_transform(p_type)
    type_int = type_int.reshape(len(type_int), 1)
    # Uncomment for house types
    # print(list(lab_enc.classes_))
    # print(type_int)
    oh_enc = OneHotEncoder(sparse=False)
    type_enc = oh_enc.fit_transform(type_int)

    # scaler = StandardScaler()
    # income = (income-income.min())/(income.max()-income.min())
    # area = (area-area.min())/(area.max()-area.min())
    # beds = (beds-beds.min())/(beds.max()-beds.min())
    # baths = (baths-baths.min())/(baths.max()-beds.min())

    X = np.vstack((income, beds, baths, area))
    X = np.transpose(X)
    # X = scaler.fit_transform(X)

    X = np.hstack((X, type_enc)) # When excluded, tends to give a better R2 score
    print(X)

    # poly = PolynomialFeatures(2)
    # X = poly.fit_transform(X)


    # Split training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Linear Regression only use cross val for mean squared error and std_error, doesn't use C
    # C value cross validation
    mean_error = []
    std_error = []
    model = linear_model.LinearRegression()
    temp = []
    kf = KFold(n_splits=5)
    # Divide data into train and test
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        # Calculate errors
        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

    # C value cross validation
    mean_error = []
    std_error = []
    # C values to validate
    CL_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]  # Loop for each value
    for Ci in CL_range:
        model = linear_model.Lasso(alpha=1/(2*Ci), max_iter=10000)
        temp = []
        kf = KFold(n_splits=5)
        # Divide data into train and test
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            # Calculate errors
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    # Plot mean square error vs C value
    plt.errorbar(CL_range, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('MSE')
    plt.title('Lasso Regression - C Cross Validation')
    plt.xscale("log")
    plt.show()

    # C value cross validation
    mean_error = []
    std_error = []
    # C values to validate
    CL_range = [0.1, 1, 10, 100, 1000]  # Loop for each value
    for Ci in CL_range:
        model = linear_model.Ridge(alpha=1/(2*Ci), max_iter=1000)
        temp = []
        kf = KFold(n_splits=5)
        # Divide data into train and test
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], ypred))
        # Calculate errors
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    # Plot mean square error vs C value
    plt.errorbar(CL_range, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('MSE')
    plt.title('Ridge Regression - C Cross Validation')
    plt.xscale("log")
    plt.show()

    # Cross validation - KNN k
    mean_error = []
    std_error = []
    # C values to validate
    K_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Loop for each k
    for Ki in K_range:
        model = KNeighborsRegressor(n_neighbors=Ki, weights='uniform')
        temp = []
        kf = KFold(n_splits=5)
        # Divide data into train and test
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], ypred))
        # Calculate errors
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    # Plot mean square error vs C value
    plt.errorbar(K_range, mean_error, yerr=std_error)
    plt.xlabel('Ki')
    plt.ylabel('MSE')
    plt.xlim((0, 10))
    plt.title('kNN - k Cross Validation')
    plt.show()

    # Create Models
    # Linear Regression
    linear_mod = linear_model.LinearRegression()
    linear_mod.fit(X_train, y_train)
    print_evaluation(linear_mod, "Linear Regression", X_train, X_test, y_train, y_test)

    # Lasso Regression
    lasso_model = linear_model.Lasso(alpha=1/(2*LASSO_C_VALUE), max_iter=10000)
    lasso_model.fit(X_train, y_train)
    print_evaluation(lasso_model, "Lasso Regression", X_train, X_test, y_train, y_test)

    # Ridge Regression
    ridge_model = linear_model.Ridge(alpha=1/(2*RIDGE_C_VALUE), max_iter=10000)
    ridge_model.fit(X_train, y_train)
    print_evaluation(ridge_model, "Ridge Regression", X_train, X_test, y_train, y_test)

    # kNN Regression
    kNN_model = KNeighborsRegressor(n_neighbors=K_VALUE, weights='uniform')
    kNN_model.fit(X_train, y_train)
    print_evaluation(kNN_model, "KNN", X_train, X_test, y_train, y_test)

    # Baseline - Predicts the average value
    base_ypred = [y_train.mean()] * len(y_test)
    base_MSE = mean_squared_error(y_test, base_ypred)
    base_R2 = r2_score(y_test, base_ypred)
    print(f"Base Model Test >> MSE = {round(base_MSE, 4)}, R2 = {round(base_R2, 4)}")

    X_train2 = zip(*X_train)
    X_train2 = list(X_train2)

    X_test2 = zip(*X_test)
    X_test2 = list(X_test2)

    # TODO not sure on this plot?
    # Plot predictions
    # 0 for income, 1 for besds, 2 for baths, 3 for area
    input = 1
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(X_train2[input], y_train, color='green', marker='+', label="Training data")
    plt.scatter(X_test2[input], linear_mod.predict(X_test), color='red', marker='x', label="Linear Model")
    plt.scatter(X_test2[input], lasso_model.predict(X_test), color='yellow', marker='o', label="Lasso Model")
    plt.scatter(X_test2[input], ridge_model.predict(X_test), color='black', marker='*', label="Ridge Model")
    plt.scatter(X_test2[input], kNN_model.predict(X_test), color='blue', marker='D', label="KNN model")
    plt.legend()
    plt.xlabel("Average Income")
    plt.ylabel("House Price")
    plt.show()

def print_evaluation(model, model_type, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    R2_train = r2_score(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    R2_test = r2_score(y_test, y_pred_test)
    print(f"{model_type} Train >> MSE = {round(MSE_train, 4)}, R2 = {round(R2_train, 4)}")
    print(f"{model_type} Test  >> MSE = {round(MSE_test, 4)}, R2 = {round(R2_test, 4)}")


def normalise(data_array: np.array) -> np.array:
    norm = np.linalg.norm(data_array)
    normalized_array = data_array/norm
    return normalized_array


if __name__ == "__main__":
    main('ml-dataset.csv')
