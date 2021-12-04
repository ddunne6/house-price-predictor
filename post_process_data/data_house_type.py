import numpy as np
import pandas as pd
import csv

def main(data):
    df = pd.read_csv(data)
    property_type = df.iloc[:, 6]
    diff_property_types = []
    ##check types
    for property in property_type:
        if property not in diff_property_types:
            diff_property_types.append(property)

    print(diff_property_types)
    ##create new types
    house = ['Terraced House ', 'Semi-Detached House ', 'End of Terrace House ', 
    'Detached House ', 'House ', 'Cottage ', 'Bungalow ', 'Townhouse ', 'Period House ', 
    'Country House ', 'Live-Work Unit ', 'Dormer ', 'Mews ', 'Holiday Home ']
    apartment= ['Apartment ', 'Penthouse ', 'Studio ', 'Duplex ']
    other=['Farm ', 'Site ', 'Investment Property ']

    ##remove unwanted types
    df = df[~df['Type'].isin(other)]

    ##update type categories
    for i, property in enumerate(df['Type']):

        if property in house:
            df.iloc[i, 6] = 'House'
        elif property in apartment:
            df.iloc[i, 6] = 'Apartment'

    print(df)
    df.to_csv('categorised-ml-dataset.csv')

if __name__ == "__main__":
    main('ml-dataset.csv')