import numpy as np
import pandas as pd
from random import random


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
    print("[x] Generating training examples...")
    flow = np.loadtxt(data_path)
    pdata = pd.DataFrame({"a": flow})
    return train_test_split(pdata)
