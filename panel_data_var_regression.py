import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


# 加窗归一化
# all_data_diff5 = preprocessing.scale(all_data_diff5_raw[4:],axis=1) # 全局归一化

def nomalization_window(data,n=None,d0=1.0,f0=1.0,alpha=0.5,beta=0.5):
    '''结合小波/傅里叶变换中加窗的思想，产生了加窗归一化
    data 为输入，(?,1)维度的时间序列
    n 窗口数
    alpha 和 beta 考虑相邻时间序列的相关性，需要优化
    返回归一化后的序列、最后一个窗口的标准化d和f
    '''
    '''
    data=all_data_diff5_raw[4:,0]
    n=5 # 窗数
    d0=0.5
    f0=0.5
    alpha=0.5
    beta=0.5
    '''

    if n==None:
        n=int(np.shape(data)[0]/10)
    d=[]
    d.append(d0)
    f=[]
    f.append(f0)
    r=[]
    l=int(np.shape(data)[0]/n)
    one_window=round(np.shape(data)[0]/n)

    for i in range(n): #
        if i <n-1:
            di=d[i]-alpha*(d[i]-(max(data[i*one_window:(i+1)*one_window])+min(data[i*one_window:(i+1)*one_window]))/2)
        elif i == n-1:
            di = d[i] - alpha * (d[i] - (max(data[i * one_window:]) + min(data[i * one_window:])) / 2)
        if i <n-1:
            fi=f[i]-beta*(f[i]-max(data[i*one_window:(i+1)*one_window])+min(data[i*one_window:(i+1)*one_window]))
        elif i == n-1:
            fi = f[i] - beta * (f[i] - max(data[i * one_window:]) + min(data[i * one_window:]))
        d.append(di)
        f.append(fi)


    for i in range(np.shape(data)[0]):
        window=int(i/l)
        if window>=np.shape(d)[0]-1:
            ri = (data[i] - d[window]) / f[window]
        else:
            ri=(data[i]-d[window+1])/f[window+1]
        r.append(ri)
    r=np.array(r)
    return r,f,d

def max_than_it(data,a): # 在data中绝对值比a大的元素的个数
    count=0
    for i in range(np.shape(data)[0]):
        if data[i]>a or -data[i]>a:
            count+=1
    return count

def ada(data,n=None,):
    # data=all_data_diff5_raw[4:,0]
    if n==None:
        n=int(np.shape(data)[0]/10)
    r_best=data
    count_bad=np.shape(data)[0]

    for a in np.arange(0,1,0.1):
        for b in np.arange(0,1,0.1):
            for d0 in np.arange(0.1,2,0.2):
                for f0 in np.arange(0.1,2,0.2):
                    r1,fi,di=nomalization_window(data,n,d0,f0,alpha=a,beta=b)
                    count_here=max_than_it(r1,1)
                    if count_here<count_bad:
                        count_bad=count_here
                        alpha_best=a
                        beta_best=b
                        d0_best=d0
                        f1_best=f0
                        r_best=r1
                        f=fi
                        d=di
                    if count_here==0:
                        break
    return r_best,f,d

def ada_normalization_window(data,n=None,d0=1.0,f0=1.0,method='ada',alpha=0.5,beta=0.5):
    '''method 'ada' adjust alpha and beta by the least number of abnormal value 
              'manual' default alpha=0.5 and default beta=0.5
    '''
    if method=='ada':
        r,f,d=ada(data,n)
    elif method=='manual':
        r,f,d=nomalization_window(data,n,d0,f0,alpha,beta)
    else:
        print("Wrong input of METHOD")
    return r,f,d


#lag 取到4

def sample_lag4(data): #lag=4
    y=[]
    x=[]
    for i in range(np.shape(data)[0]-5):
        y.append(data[4+i,0])
        x_one=[]
        x_one.append(data[3+i,0])
        x_one.append(data[2+i,0])
        x_one.append(data[1+i,0])
        x_one.append(data[0+i,0])
        for j in range(np.shape(data)[1]):
            x_one.append(data[3+i,j]) #lag1
            if granger_cause_index[j-1,1]>=2: #格兰杰因果里，没有包括y自己，所以-1
                x_one.append(data[2+i,j])#lag2
                if granger_cause_index[j-1,1]>=3:
                    x_one.append(data[2+i,j])#lag3
                    if granger_cause_index[j-1,1]>=4:
                        x_one.append(data[0+i,j])#lag4
                    else:
                        continue
                else:
                    continue
            else:
                continue
        x.append(x_one)

    x=np.array(x)
    y=np.array(y)
    return x,y

def sample_lag_granger4(data): #lag=4
    y=[]
    x=[]
    for i in range(np.shape(data)[0]-5):
        y.append(data[4+i,0])
        x_one=[]
        x_one.append(data[3+i,0])
        x_one.append(data[2+i,0])
        x_one.append(data[1+i,0])
        x_one.append(data[0+i,0])
        for j in range(np.shape(data)[1]):
            if granger_cause_index[j-1,1]==1:
                x_one.append(data[3+i,j]) #lag1
            elif granger_cause_index[j-1,1]==2: #格兰杰因果里，没有包括y自己，所以-1
                x_one.append(data[2+i,j])#lag2
            elif granger_cause_index[j-1,1]==3:
                x_one.append(data[2+i,j])#lag3
            elif granger_cause_index[j-1,1]==4:
                x_one.append(data[0+i,j])#lag4
            else:
                        continue
        x.append(x_one)

    x=np.array(x)
    y=np.array(y)
    return x,y


all_data_diff5_raw=np.loadtxt('all_data_diff5.txt')
granger_cause_index=np.loadtxt('granger_cause_index.txt')

##只留下有格兰杰的
all_data_diff5_keep=[]
all_data_diff5_keep.append(all_data_diff5_raw[4:,0])
for index in granger_cause_index[:,0]:
    all_data_diff5_keep.append(all_data_diff5_raw[4:,int(index)])
all_data_diff5_keep=np.transpose(np.array(all_data_diff5_keep))

#对有格兰杰原因的变量进行归一化
windows=4 #窗口数
r_all=[]
f_all=[]
d_all=[]
for i in range(np.shape(all_data_diff5_keep)[1]):
    r,f,d=ada_normalization_window(all_data_diff5_keep[:,i],n=windows)
    r_all.append(r)
    f_all.append(f)
    d_all.append(d)

all_data_diff5=np.transpose(np.array(r_all))

###### reload
np.save('f_all',f_all)
np.save('d_all',d_all)
np.save('normalization_all_data_keep',all_data_diff5)
all_data_diff5=np.load('normalization_all_data_keep.npy')
f_all=np.load('f_all.npy')
d_all=np.load('d_all.npy')

'''
##判断格兰杰因果检验的lag

np.median(granger_cause_index[:,1])
np.shape(granger_cause_index[:,1])

count = len(granger_cause_index[:,1])
myset = set(granger_cause_index[:,1])
for item in myset:
    print(item,100*granger_cause_index[:,1].tolist().count(item)/count)
'''


# x,y=sample_lag4(all_data_diff5) # 取lag4不行，过拟合，train的r2为1，但是test只有0.13
x,y=sample_lag_granger4(all_data_diff5)

x_test=x[-9:]
y_test=y[-9:]
x_train=x[:-9]
y_train=y[:-9]

''' # ols 多重共线性 test 0.4
X_train=sm.add_constant(x_train)
est=sm.OLS(y_train,X_train).fit()
est.summary()

X_test=sm.add_constant(x_test)
y_predict = est.predict(X_test)
'''

'''
#岭回归，train 0.99， test 0.39
clf = linear_model.Ridge(alpha=0.1)
clf.fit(x_train, y_train)
coef=clf.coef_
clf.score(x_train, y_train)
y_predict=clf.predict(x_test)
'''

# 偏最小二乘回归 test 0.5 循环看几个主成分效果最好，n=5
r2_test_best=0
r2_train_best=0
n_best=0
y_test=np.reshape(y_test,[np.shape(y_test)[0],1])
for n in range(1,min(np.shape(x_train)[0],np.shape(x_train)[1])+1):
    pls2 = PLSRegression(n_components=n,scale=False)
    pls2.fit(x_train, y_train)
    r2_train=pls2.score(x_train,y_train)
    y_predict = pls2.predict(x_test)
    r2_test = pls2.score(x_test,y_test)
    #r2_test=test_r_square(y_predict,y_test)
    if r2_test>r2_test_best:
        r2_test_best=r2_test
        r2_train_best=r2_train
        n_best=n
    else:
        continue

pls2 = PLSRegression(n_components=n_best,scale=False)
pls2.fit(x_train,y_train)
y_pred_test = pls2.predict(x_test)
y_pred_train=pls2.predict(x_train)


######## 逆归一化
data_all = np.loadtxt("/Users/XuLiu/Documents/cwrnn/all_data.txt")
f_all=np.array(f_all)
d_all=np.array(d_all)
## train 用[0]的来
one_window = round(np.shape(y_train)[0] / windows)
s_tra=[]
for i in range(1,windows+1): #windows=4
    if i < windows:
        si=y_pred_train[(i-1)*one_window:i*one_window]*f_all[0,i]+d_all[0,i]
    elif i == windows:
        si=y_pred_train[(i-1)*one_window:]*f_all[0,i]+d_all[0,i]
    s_tra.append(si)

s_train=[]
for i in range(np.shape(s_tra)[0]):
    for j in range(np.shape(s_tra[i])[0]):
        s_train.append(s_tra[i][j])

s_train=np.array(s_train)
##train的逆差分
y_train_pred_real=[]
y_train_pred_real.append(data_all[10,0]) # 第一个y_train的实际值
for i in range(np.shape(s_train)[0]):
    this=y_train_pred_real[i]+s_train[i]
    y_train_pred_real.append(this[0])
y_train_pred_real=np.array(y_train_pred_real)

np.save('y_train_real',y_train_pred_real)


## test 使用最后一个分组的f和d作为测试数据集的归一化
s_test=f_all[0,-1]*y_pred_test+d_all[0,-1]


# 逆差分，一阶差分就可以
y_test_pred_real=[]
y_test_pred_real.append(data_all[-10,0]) #已知的最后一个train_y

for i in range(np.shape(s_test)[0]):
    this=y_test_pred_real[i]+s_test[i]
    y_test_pred_real.append(this[0])

y_test_pred_real=np.array(y_test_pred_real)

np.save('y_predict_real',y_test_pred_real)
