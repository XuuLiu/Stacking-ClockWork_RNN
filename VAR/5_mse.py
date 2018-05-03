import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

pan_train_pred=np.load('/Users/XuLiu/paneldata/y_train_real_ols2.npy') #26个
pan_valid_pred=np.load('/Users/XuLiu/paneldata/y_predict_real_ols2.npy')
data_all = np.loadtxt("/Users/XuLiu/Documents/cwrnn/all_data.txt")
y_real=data_all[1:-8,0] # lag90


loss_train=math.sqrt(mean_squared_error(y_real[:26],pan_train_pred)) #1.2267774755475551



loss_validation=math.sqrt(mean_squared_error(y_real[-10:],pan_valid_pred)) #1.2890196304786823

