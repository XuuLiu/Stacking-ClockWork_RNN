import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stt
import seaborn as sns



all_data_diff5=np.loadtxt('all_data_diff5.txt')

cointegration=[]
for i in range(1,np.shape(all_data_diff5)[1]-1):
    result = stt.coint(all_data_diff5[4:,0], all_data_diff5[4:,i])
    if result[1]<=0.05:
        cointegration.append([i,result[1]])
# No cointegration 