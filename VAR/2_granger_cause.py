import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


all_data_diff5=np.loadtxt('all_data_diff5.txt')

granger_cause_index=[]
for i in range(1,np.shape(all_data_diff5)[1]-1):
    data=[]
    data.append(all_data_diff5[4:,0])
    data.append(all_data_diff5[4:,i])
    data=np.transpose(np.array(data))
    result=grangercausalitytests(data,12,True)
    for j in range(1,13):
        if result[j][0]['params_ftest'][1]<=0.05:#step # no use # params_ftest # p_value<0.05
            granger_cause_index.append([i,j]) # index of x , lag
            break

granger_cause_index=np.array(granger_cause_index)

np.savetxt('granger_cause_index.txt',granger_cause_index)

