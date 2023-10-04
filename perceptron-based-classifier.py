import numpy as np
import pandas as pd
import math
import time

def perceptron_based_classifier(df, activation_function, epsilon):
    # initialize variables
    ite = 5000
    num_patterns = df.shape[0]
    ni = 3
    alpha = 0.01
    ww = [np.random.uniform(-0.5, 0.5),np.random.uniform(-0.5, 0.5),np.random.uniform(-0.5, 0.5)]
    # change the dataframe to a NumPy array
    pat = df.to_numpy()
    # since the 3rd column is the label for the type of car, it's our desired output
    dout = np.copy(pat[:, 2])
    dout[dout == 0] = -1

    # since we saved the dout, change the 3rd column into all 1's so it will always include bias
    pat[:, 2] = 1
    
    
    for iteration in range(0, ite):
        ou = np.zeros(num_patterns)
        error_count = 0
        for pattern in range(0, num_patterns):
            net = 0
            for i in range(0, ni):
                net = net + ww[i] * pat[pattern][i]
            
            if activation_function == 1:
                ou[pattern] = np.sign(net)
                err = dout[pattern] - ou[pattern]
    
            if activation_function == 0:
                ou[pattern] = fbip(net)
                err = dout[pattern] - ou[pattern]; 
            
            learn = alpha * err
           # print(iteration, pattern, net, err, learn, ww)
            for j in range(0, ni-1):
                ww[j] = ww[j] + learn * pat[pattern][j]
            ww[ni-1] =  ww[ni-1] + learn

        error_count = 0
        for pattern in range(0, num_patterns):
            net = 0
            for i in range(0, ni-1):
                net = net + (ww[i] * pat[pattern][i])
            net = net + ww[ni-1]
            if net > 0:
                prediction = 1
            else:
                prediction = -1
            if prediction != dout[pattern]:
                error_count += 1

        # print(error_count)
        if error_count < epsilon:
            break
    print(ww)
    return ww

def fbip(net):
    k = .4
    return 2 / (1 + math.exp(-2*k*net)) - 1

def confusion_matrix(data, ww):
    df = parse_data(data)
    pat = df.to_numpy()
    num_patterns = 4000
    ni = 3
    dout = np.copy(pat[:, 2])
    dout[dout == 0] = -1
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    for pattern in range(0, num_patterns):
        net = 0
        for i in range(0, ni-1):
            net = net + (ww[i] * pat[pattern][i])
        net = net + ww[ni-1]
        if net > 0:
            prediction = 1
        else:
            prediction = -1
        
        if dout[pattern] == 1 and prediction == 1:
            TP += 1
        elif dout[pattern] == -1 and prediction == 1:
            FP += 1
        elif dout[pattern] == 1 and prediction == -1:
            FN += 1
        elif dout[pattern] == -1 and prediction == -1:
            TN += 1
    return TP, FP, FN, TN



def normalize_data(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def parse_data(data):
    df = pd.read_csv(data, header=None, names = ['Price(USD)', 'Weight(lbs)', 'Desired_Output'])
    return normalize_data(df)

def hard_activation_function(data, epsilon, percent=100):
    df = parse_data(data)
    if percent == 75:
        df = random75(data)

    if percent == 25:
        df = random25(data)

    return perceptron_based_classifier(df,1, epsilon)

def soft_activation_function(data, epsilon, percent=100):
    df = parse_data(data)
    if percent == 75:
        df = random75(df)

    if percent == 25:
        df = random25(df)

    return perceptron_based_classifier(df,0, epsilon)

def random75(df):
    sample = df.sample(frac=0.75)
    return sample

def random25(df):
    sample = df.sample(frac=0.25)
    return sample

if __name__ == '__main__':
    print(confusion_matrix('groupC.txt', soft_activation_function('groupC.txt', 700, 75)))
    