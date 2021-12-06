import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(dataset):
    df = pd.read_csv(dataset)
    print(df.head())
    y = df.iloc[:, 1]
    income = df.iloc[:, 2]
    beds = df.iloc[:, 3]
    baths = df.iloc[:, 4]
    area = df.iloc[:, 5]
    p_type = df.iloc[:, 6]

    plt.style.use('ggplot')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(income, y, color='r', marker='o', label="training data")
    plt.xlabel('Median Income of Area (€)')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.title(f"Median Income of Area Vs Price of House")
    plt.show()

    plt.scatter(area, y, color='b', marker='+', label="training data")
    plt.xlabel('Area of House (m^2)')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.xlim(0, 1000)
    plt.title(f"Area of House Vs Price of House")
    plt.show()

    plt.scatter(p_type, y, color='c', marker='1', label="training data")
    plt.xlabel('Type of House')
    plt.ylabel('House Price (€)')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title(f"Type of House Vs Price of House")
    plt.show()

    plt.scatter(beds, y, color='r', marker='o', label="beds")
    plt.scatter(baths, y, color='g', marker='x', label="baths")
    plt.xlabel('Number of beds/baths')
    plt.ylabel('House Price (€)')
    plt.legend()
    plt.xlim(0, 16)
    plt.title(f"Number of Units Vs Price of House")
    plt.show()


if __name__ == "__main__":
    main('ml-dataset.csv')