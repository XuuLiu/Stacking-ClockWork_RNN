import numpy as np
import statsmodels.api as sm
from operator import itemgetter

def test_r_square(y_predict,y_test):
    residuals = y_predict - y_test
    SSE = np.sum((residuals)**2)
    ybar = np.mean(y_test)
    RSS = np.sum((ybar-y_predict)**2)
    TSS = np.sum((ybar-y_test)**2)
    R2 = RSS/TSS
    return R2

###train
cw_train=np.load('./pred_training/predict_y_4320.npy')
cw_train_all=cw_train.reshape([np.shape(cw_train)[0]*np.shape(cw_train)[1],1])
np.savetxt('./cw_train',cw_train_all)

all_date=np.loadtxt('./date.txt')

data_all = np.loadtxt("/Users/XuLiu/Documents/cwrnn/all_data.txt")
y_real=data_all[1:-8,0] # lag90

train_date=all_date[90:90+np.shape(cw_train_all)[0]]

cw_train_all_avg=[]
years=sorted(set(train_date[:,0]))
for year in years:
    q1=[]
    q2=[]
    q3=[]
    q4=[]
    for i in range(np.shape(cw_train_all)[0]):
        if train_date[i,0]==year and train_date[i,1] in [1,2,3]:
            q1.append(cw_train_all[i])
        elif train_date[i,0]==year and train_date[i,1] in [4,5,6]:
            q2.append(cw_train_all[i])
        elif train_date[i,0]==year and train_date[i,1] in [7,8,9]:
            q3.append(cw_train_all[i])
        elif train_date[i,0]==year and train_date[i,1] in [10,11,12]:
            q4.append(cw_train_all[i])
        else:
            continue
    if q1 !=[]:
        cw_train_all_avg.append([year,1,np.mean(q1)])
    if q2 !=[]:
        cw_train_all_avg.append([year,2,np.mean(q2)])
    if q3 !=[]:
        cw_train_all_avg.append([year,3,np.mean(q3)])
    if q4 != []:
        cw_train_all_avg.append([year, 4, np.mean(q4)])

cw_train_all_avg=np.array(sorted(cw_train_all_avg,key=itemgetter(0,1)))
#panel data result
pan_train_pred=np.load('/Users/XuLiu/pennaldata/y_train_real.npy') #26个


x=zip(cw_train_all_avg[-26:,2],pan_train_pred)
x=np.array(x)
x=sm.add_constant(x)
est=sm.OLS(y_real[-26:],x).fit()
est.summary()



#validation
cw_pred=np.load('./pred_validation/predict_y_4320.npy')
pan_pred=np.load('/Users/XuLiu/pennaldata/y_predict_real.npy')

cw_pred_all=cw_pred.reshape([np.shape(cw_pred)[0]*np.shape(cw_pred)[1],1])

cw_pred_avg=[]
cw_pred_avg.append(np.mean(cw_pred_all[1:62])) # 2015Q4
cw_pred_avg.append(np.mean(cw_pred_all[62:121])) # 2016Q1
cw_pred_avg.append(np.mean(cw_pred_all[121:182])) # 2016Q2
cw_pred_avg.append(np.mean(cw_pred_all[182:246])) # 2016Q3
cw_pred_avg.append(np.mean(cw_pred_all[246:306])) # 2016Q4
cw_pred_avg.append(np.mean(cw_pred_all[306:365])) # 2017Q1
cw_pred_avg.append(np.mean(cw_pred_all[365:425])) # 2017Q2
cw_pred_avg.append(np.mean(cw_pred_all[425:490])) # 2017Q3
#cw_pred_avg.append(np.mean(cw_pred_all[490:])) # 2017Q4
cw_pred_avg=np.array(cw_pred_avg)

y_real_test=data_all[-8:,0]

x_test=np.array(zip(cw_pred_avg,pan_pred))
# 不应该用这个fit，应该用训练的结果fit

y_test_pred=0.878*x_test[:,0]+0.7672#-0.05*x_test[:,1]

#这个算法有点问题，，R2，再搞一下，另外那个project也再搞一下这个
R2_test=test_r_square(y_predict=y_test_pred,y_test=y_real_test)
