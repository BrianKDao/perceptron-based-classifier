import numpy as np
import pandas as pd
import time

def perceptron_based_classifier(df):
    # initialize variables
    iterations = 5000
    number_of_patterns = 4000
    number_of_inputs = 3
    alpha = 0.0006
    ww = [np.random.uniform(-0.5, 0.5),np.random.uniform(-0.5, 0.5),np.random.uniform(-0.5, 0.5)]
    # change the dataframe to a NumPy array
    patterns = df.to_numpy()
    # since the 3rd column is the label for the type of car, it's our desired output
    dout = np.copy(patterns[:, 2])
    # since we saved the dout, change the 3rd column into all 1's so it will always include bias
    patterns[:, 2] = 1
    for iteration in range(0, iterations):
        output = np.zeros(number_of_patterns)
        error_count = 0
        for pattern in range(0, number_of_patterns):
            net = 0
            for i in range(0, number_of_inputs):
                net = net + ww[i] * patterns[pattern][i]
            # unipolar output
            output[pattern] = (np.sign(net) + 1) / 2 
            err = dout[pattern] - output[pattern]
            if err != 0:
                error_count += 1
            learn = alpha * err
            print(iteration, pattern, net, err, learn, ww)
            for i in range(0, number_of_inputs):
                ww[i] = ww[i] + learn * patterns[pattern][i]
        if error_count < 0.0005:
            break

      
def normalize_data(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    return df

def parse_data(data):
    df = pd.read_csv(data, header=None, names = ['Price(USD)', 'Weight(lbs)', 'Desired_Output'])
    return normalize_data(df)

if __name__ == '__main__':
    print(perceptron_based_classifier(parse_data('groupA.txt')))