import numpy as np
import pandas as pd
from random import random

# Just a random example of sequences to train the Clockwork RNN.
# http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/


def _load_data(data, n_prev=90): #n_prev is how many steps forward
    # data should be pd.DataFrame()
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def train_test_split(df): #, test_size=0.2):
    ntrn = 2014 #int(round(len(df) * (1 - test_size)))
    X_all, y_all = _load_data(df)
    X_train=X_all[0:ntrn+1] #2015=35*55+90
    y_train =y_all[0:ntrn+1]
    X_test=X_all[ntrn:]
    y_test = y_all[ntrn:] # 550= 55*10
    return (X_train, y_train), (X_test, y_test)


def generate_data(data_path):
    #data_path="/Users/XuLiu/Documents/cwrnn/price_avg.txt"
    #num_examples=1000
    print("[x] Generating training examples...")
    flow = np.loadtxt(data_path)
    #flow = (list(range(1, 10, 1)) + list(range(10, 1, -1))) * num_examples
    #pdata = pd.DataFrame({"a": flow, "b": flow})
    pdata = pd.DataFrame({"a": flow})
    #pdata.b = pdata.b.shift(9) # b index shift 9 foreword
    #data = pdata.iloc[10:] * random()  # some noise
    return train_test_split(pdata)