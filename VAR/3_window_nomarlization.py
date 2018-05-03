import numpy as np
from sklearn.cross_decomposition import PLSRegression



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


###不加窗的

def max_min_normal(data): #test和train的划分是到-9，计算min和MAX只用train的
    norm=[]
    minmax=[]
    for i in range(np.shape(data)[1]):
        min_colm=min(data[:-9,i])
        max_colm=max(data[:-9,i])
        norm.append((data[:-9,i]-min_colm)/(max_colm-min_colm))
        minmax.append([min_colm,max_colm])
    return np.transpose(np.array(norm)),minmax

all_data_diff5,minmax=max_min_normal(all_data_diff5_keep)




###### reload
np.save('f_all',f_all)
np.save('d_all',d_all)
np.save('normalization_all_data_keep',all_data_diff5) #加窗标准化

np.save('normalization_all_data_keep_minmax',all_data_diff5) #整体标准化
np.save('normalization_minmax',minmax)




