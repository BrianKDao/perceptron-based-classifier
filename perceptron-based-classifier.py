import numpy as np
import pandas as pd

def normalize_data(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    return df

def parse_data(data):
    df = pd.read_csv(data, header=None, names = ['Price(USD)', 'Weight(lbs)', 'Type'])
    return normalize_data(df)

if __name__ == '__main__':
    print(parse_data('groupA.txt')) 