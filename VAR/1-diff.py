import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.tsa.stattools import adfuller
import math



data_all = np.loadtxt("/Users/XuLiu/Documents/cwrnn/all_data.txt")

##Mann-Whitney U test
#results = np.zeros((data_all.shape[1],2))
#for i in range(data_all.shape[1]):
#    u, prob = mannwhitneyu(data_all[:,i], data_all[:,0])
#    results[i,:] = u, prob

#unit root test
def unit_root_test(data,diff):
    # return non stationary time series' index
#    data=all_data_diff1
    non_stationary = []
    for i in range(np.shape(data)[1]):
        u, prob = adfuller(data[diff:, i])[0:2]
        if prob>0.05:
            non_stationary.append([i,u,prob])
    return np.array(non_stationary)

non_stationary_diff0=unit_root_test(data_all,0)
pdata_all=pd.DataFrame(data_all)
# diff1
all_data_diff1=[]
index_diff1=[]
for i in range(np.shape(pdata_all)[1]):
    if i in non_stationary_diff0[:,0]:
        all_data_diff1.append(pdata_all.iloc[:,i].diff(1))
        index_diff1.append(True)
    else:
        all_data_diff1.append(pdata_all.iloc[:,i])
        index_diff1.append(False)
all_data_diff1=np.transpose(np.array(all_data_diff1))
non_stationary_diff1=unit_root_test(all_data_diff1[:,index_diff1],1)



#diff2
all_data_diff2=[]
index_diff2=[]
all_data_diff1=pd.DataFrame(all_data_diff1)
for i in range(np.shape(all_data_diff1)[1]):
    if i in non_stationary_diff1[:,0]:
        all_data_diff2.append(all_data_diff1.iloc[:,i].diff(1))
        index_diff2.append(True)
    else:
        all_data_diff2.append(all_data_diff1.iloc[:,i])
        index_diff2.append(False)
all_data_diff2=np.transpose(np.array(all_data_diff2))
non_stationary_diff2=unit_root_test(all_data_diff2[:,index_diff2],2)


#diff3
all_data_diff3=[]
index_diff3=[]
all_data_diff2=pd.DataFrame(all_data_diff2)
for i in range(np.shape(all_data_diff2)[1]):
    if i in non_stationary_diff2[:,0]:
        all_data_diff3.append(all_data_diff2.iloc[:,i].diff(1))
        index_diff3.append(True)
    else:
        all_data_diff3.append(all_data_diff2.iloc[:,i])
        index_diff3.append(False)
all_data_diff3=np.transpose(np.array(all_data_diff3))
non_stationary_diff3=unit_root_test(all_data_diff3[:,index_diff3],3)


#diff4
all_data_diff4=[]
index_diff4=[]
all_data_diff3=pd.DataFrame(all_data_diff3)
for i in range(np.shape(all_data_diff3)[1]):
    if i in non_stationary_diff3[:,0]:
        all_data_diff4.append(all_data_diff3.iloc[:,i].diff(1))
        index_diff4.append(True)
    else:
        all_data_diff4.append(all_data_diff3.iloc[:,i])
        index_diff4.append(False)
all_data_diff4=np.transpose(np.array(all_data_diff4))
non_stationary_diff4=unit_root_test(all_data_diff4[:,index_diff4],4)

#diff5
all_data_diff5=[]
index_diff5=[]
all_data_diff4=pd.DataFrame(all_data_diff4)
for i in range(np.shape(all_data_diff4)[1]):
    if i in non_stationary_diff4[:,0]:
        all_data_diff5.append(all_data_diff4.iloc[:,i].diff(1))
        index_diff5.append(True)
    else:
        all_data_diff5.append(all_data_diff4.iloc[:,i])
        index_diff5.append(False)
all_data_diff5=np.transpose(np.array(all_data_diff5))
non_stationary_diff5=unit_root_test(all_data_diff5[:,index_diff5],5)

np.savetxt('all_data_diff5.txt',all_data_diff5)


all_data_diff5=np.loadtxt('all_data_diff5.txt')
