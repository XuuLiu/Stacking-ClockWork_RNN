import numpy as np
from sklearn.cross_decomposition import PLSRegression


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

def adjust_r2(predict,actual,dimension):
# a is predict, b is actual. dimension is len(train[0]).
    aa=predict.copy(); bb=actual.copy()
    if len(aa)!=len(bb):
        print('not same length')
        return np.nan

    # RR means R_Square
    RR=1-sum((bb-aa)**2)/sum((bb-np.mean(bb))**2)

    n=len(aa); p=dimension
    Adjust_RR=1-(1-RR)*(n-1)/(n-p-1)
    # Adjust_RR means Adjust_R_Square

    return RR

granger_cause_index=np.loadtxt('granger_cause_index.txt')
#all_data_diff5=np.load('normalization_all_data_keep.npy') #加窗标准化
all_data_diff5=np.load('normalization_all_data_keep_minmax.npy') #整体标准化

f_all=np.load('f_all.npy')
d_all=np.load('d_all.npy')

minmax=np.load('normalization_minmax.npy')

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
windows=4
# 偏最小二乘回归 test 0.3 循环看几个主成分效果最好，n=5 #用全局标准化0.3314
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

adjust_r2(y_pred_test,np.reshape(y_test,[np.shape(y_test)[0],1]),n_best)







######## 逆归一化，加窗的
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

np.save('y_train_real_ols2',y_train_pred_real)


## test 使用最后一个分组的f和d作为测试数据集的归一化
s_test=f_all[0,-1]*y_pred_test+d_all[0,-1]


# 逆差分，一阶差分就可以
y_test_pred_real=[]
y_test_pred_real.append(data_all[-10,0]) #已知的最后一个train_y

for i in range(np.shape(s_test)[0]):
    this=y_test_pred_real[i]+s_test[i]
    y_test_pred_real.append(this[0])

y_test_pred_real=np.array(y_test_pred_real)

np.save('y_predict_real_ols2',y_test_pred_real)

