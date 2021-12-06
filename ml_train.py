# Group 7 Machine Learning Project
# David Dunne - 17329756
# Eoin Lynch -
# Charlie Maguire - 17332641

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Hyperparameters
LASSO_C_VALUE = 0.0001
RIDGE_C_VALUE = 0.1
K_VALUE = 8

SHOW_CROSSVAL = True


def main(dataset):
    # Read in the dataset
    df = pd.read_csv(dataset)
    print(df.head())
    y = df.iloc[:, 1].to_numpy()
    income = df.iloc[:, 2].to_numpy()
    beds = df.iloc[:, 3].to_numpy()
    baths = df.iloc[:, 4].to_numpy()
    area = df.iloc[:, 5].to_numpy()
    p_type = df.iloc[:, 6].to_numpy()

    # Remove listings over a certain size
    mask = area_mask(area, 1000)
    income = income[mask]
    beds = beds[mask]
    baths = baths[mask]
    area = area[mask]
    p_type = p_type[mask]
    y = y[mask]

    # Normalise input features between 0 and 1
    income_min, income_max = np.min(income), np.max(income)
    income = normalise(income, income_min, income_max)
    beds_min, beds_max = np.min(beds), np.max(beds)
    beds = normalise(beds, beds_min, beds_max)
    baths_min, baths_max = np.min(baths), np.max(baths)
    baths = normalise(baths, baths_min, baths_max)
    area_min, area_max = np.min(area), np.max(area)
    area = normalise(area, area_min, area_max)

    # Peform one-hot encoding on property types
    lab_enc = LabelEncoder()
    type_int = lab_enc.fit_transform(p_type)
    type_int = type_int.reshape(len(type_int), 1)
    oh_enc = OneHotEncoder(sparse=False)
    type_enc = oh_enc.fit_transform(type_int)

    # Combine features to one input
    X = np.vstack((income, beds, baths, area))
    X = np.transpose(X)
    X = np.hstack((X, type_enc))

    # Split training and test data with 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    plt.rc('font', size=18)
    # Cross-Validation
    # No cross-validation for linear model - no C value to tune
    if SHOW_CROSSVAL:
        # Lasso regression C cross-validation
        mean_error = []
        std_error = []
        # C values to validate
        CL_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        for Ci in CL_range:
            model = linear_model.Lasso(alpha=1/(2*Ci), max_iter=100000)
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
        # Plot MSE vs C value
        plt.errorbar(CL_range, mean_error, yerr=std_error)
        plt.xlabel('Ci')
        plt.ylabel('MSE')
        plt.title('Lasso Regression - C Cross Validation')
        plt.xscale("log")
        plt.show()

        # Ridge regression C cross-validation
        mean_error = []
        std_error = []
        # C values to validate
        CL_range = [0.00001, 0.0001, 0.001, 0.01,
                    0.1, 1, 10, 100, 1000, 10000, 100000]
        for Ci in CL_range:
            model = linear_model.Ridge(alpha=1/(2*Ci), max_iter=10000)
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
        # Plot MSE vs C value
        plt.errorbar(CL_range, mean_error, yerr=std_error)
        plt.xlabel('Ci')
        plt.ylabel('MSE')
        plt.title('Ridge Regression - C Cross Validation')
        plt.xscale("log")
        plt.show()

        # kNN k cross-validation
        mean_error = []
        std_error = []
        # k values to validate
        K_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
        # Plot MSE vs k value
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
    print_evaluation(linear_mod, "Linear Regression",
                     X_train, X_test, y_train, y_test)

    # Lasso Regression
    lasso_model = linear_model.Lasso(alpha=1/(2*LASSO_C_VALUE), max_iter=10000)
    lasso_model.fit(X_train, y_train)
    print_evaluation(lasso_model, "Lasso Regression",
                     X_train, X_test, y_train, y_test)

    # Ridge Regression
    ridge_model = linear_model.Ridge(alpha=1/(2*RIDGE_C_VALUE), max_iter=10000)
    ridge_model.fit(X_train, y_train)
    print_evaluation(ridge_model, "Ridge Regression",
                     X_train, X_test, y_train, y_test)

    # kNN Regression
    kNN_model = KNeighborsRegressor(n_neighbors=K_VALUE, weights='uniform')
    kNN_model.fit(X_train, y_train)
    print_evaluation(kNN_model, "KNN", X_train, X_test, y_train, y_test)

    # Baseline - Predicts the average value
    base_ypred = [y_train.mean()] * len(y_train)
    base_train_MSE = mean_squared_error(y_train, base_ypred)
    base_train_R2 = r2_score(y_train, base_ypred)
    print(
        f"Base Model Training >> MSE = {round(base_train_MSE, 4)}, R2 = {round(base_train_R2, 4)}")

    base_ypred = [y_train.mean()] * len(y_test)
    base_test_MSE = mean_squared_error(y_test, base_ypred)
    base_test_R2 = r2_score(y_test, base_ypred)
    print(
        f"Base Model Test >> MSE = {round(base_test_MSE, 4)}, R2 = {round(base_test_R2, 4)}")

    # Prepare train and test data for plotting
    X_train2 = zip(*X_train)
    X_train2 = list(X_train2)
    X_test2 = zip(*X_test)
    X_test2 = list(X_test2)

    # Plot predictions
    # 0 for income, 1 for besds, 2 for baths, 3 for area
    input = 1
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(X_test2[input], y_test, color='green',
                marker='+', label="Test data")
    plt.scatter(X_test2[input], linear_mod.predict(X_test),
                color='blue', marker='D', label="Linear Model")
    plt.scatter(X_test2[input], lasso_model.predict(X_test),
                color='yellow', marker='o', label="Lasso Model")
    plt.scatter(X_test2[input], ridge_model.predict(X_test),
                color='black', marker='*', label="Ridge Model")
    plt.scatter(X_test2[input], kNN_model.predict(X_test),
                color='red', marker='x', label="KNN model")
    plt.legend()
    plt.xlabel("Average Income")
    plt.ylabel("House Price")
    plt.show()

    plt.style.use('ggplot')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=12)
    # Plot a prediction for linear regression
    # Area
    plt.scatter(denormalise(area, area_min, area_max), y,
                color='b', marker='o', label="training data")
    plt.scatter(denormalise(area, area_min, area_max), linear_mod.predict(
        X), color='r', marker='x', label="Linear Regression preds")
    plt.scatter(denormalise(area, area_min, area_max), kNN_model.predict(
        X), color='g', marker='+', label="kNN preds")
    plt.xlabel('Area of House (m^2)')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.xlim(0, 1000)
    plt.title(f"Area of House Vs Price of House")
    plt.show()

    # Plot predictions for kNN
    # Income
    plt.scatter(denormalise(income, income_min, income_max), y,
                color='b', marker='o', label="training data")
    plt.scatter(denormalise(income, income_min, income_max), kNN_model.predict(
        X), color='g', marker='x', label="predictions")
    plt.xlabel('Median Income of Area (€)')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.title(f"Median Income of Area Vs Price of House (kNN)")
    plt.show()

    # Area
    plt.scatter(denormalise(area, area_min, area_max), y,
                color='b', marker='o', label="training data")
    plt.scatter(denormalise(area, area_min, area_max), kNN_model.predict(
        X), color='r', marker='x', label="predictions")
    plt.xlabel('Area of House (m^2)')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.xlim(0, 1000)
    plt.title(f"Area of House Vs Price of House (kNN)")
    plt.show()

    # Number of Beds
    plt.scatter(denormalise(beds, beds_min, beds_max), y,
                color='b', marker='o', label="training data")
    plt.scatter(denormalise(beds, beds_min, beds_max), kNN_model.predict(
        X), color='r', marker='x', label="predictions")
    plt.xlabel('Number of beds')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.xlim(0, 16)
    plt.title(f"Number of Beds Vs Price of House (kNN)")
    plt.show()

    # Number of Bathrooms
    plt.scatter(denormalise(baths, baths_min, baths_max), y,
                color='b', marker='o', label="training data")
    plt.scatter(denormalise(baths, baths_min, baths_max), kNN_model.predict(
        X), color='g', marker='x', label="predictions")
    plt.xlabel('Number of bathrooms')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.xlim(0, 16)
    plt.title(f"Number of Bathrooms Vs Price of House")
    plt.show()


# Prints evaluation of model


def print_evaluation(model, model_type, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    R2_train = r2_score(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    R2_test = r2_score(y_test, y_pred_test)
    print(f"{model_type} Train >> MSE = {round(MSE_train, 4)}, R2 = {round(R2_train, 4)}")
    print(f"{model_type} Test  >> MSE = {round(MSE_test, 4)}, R2 = {round(R2_test, 4)}")
    if model_type != "KNN":
        print(f"{model_type} Intercept = %f" % (model.intercept_))
        print(f"{model_type} Coefficients = ", (model.coef_))


# Normalises inputs to range 0 - 1
def normalise(data_array: np.array, min, max) -> np.array:
    return (data_array - min) / (max - min)


def denormalise(norm_array: np.array, min, max) -> np.array:
    return (norm_array * (max - min) + min)


# Removes listings that include land using a mask


def area_mask(area, max_area):
    sizes = np.empty(len(area), dtype=object)
    # -1 if too large, 1 if acceptable
    for i in range(len(area)):
        if area[i] >= max_area:
            sizes[i] = -1
        else:
            sizes[i] = 1
    mask = sizes != -1
    return mask


if __name__ == "__main__":
    main('ml-dataset.csv')
