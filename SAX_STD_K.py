import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

train_path = 'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Adiac\\Adiac_TRAIN.tsv'#对于不同的数据集修改路径即可
test_path = 'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Adiac\\Adiac_TEST.tsv'#对于不同的数据集修改路径即可
train_data = np.loadtxt(train_path, delimiter='\t')
test_data = np.loadtxt(test_path, delimiter='\t')
train_label=train_data[:,0]
test_label=test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data=np.delete(test_data,0,axis=1)
train_data=train_data.astype(float)
test_data=test_data.astype(float)

w=80
alpha = 10
aOffset = ord('a')  # 字符的起始位置，从a开始
breakpoints = {'3': [-0.43, 0.43],
            '4': [-0.67, 0, 0.67],
            '5': [-0.84, -0.25, 0.25, 0.84],
            '6': [-0.97, -0.43, 0, 0.43, 0.97],
             '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
            '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
            '9':[-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
            '10':[-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]
            }
beta = breakpoints[str(alpha)]


def PLR_WFTP(data):
    high_k,low_k=0,0
    gao_x,gao_weizhi = [],[]
    di_x,di_weizhi = [],[]
    data_copy=data.copy()
    data_copy=(data-min(data))*1.0/(max(data)-min(data))
    # 1提取上下滤波点
    for i in range(1,len(data_copy)-1):
        if((data_copy[i]>data_copy[i-1]) or (data_copy[i]>data_copy[i+1])):
            gao_x.append(data_copy[i])
            gao_weizhi.append(i)
            high_k+=1
            continue
        elif((data_copy[i]<data_copy[i-1]) or (data_copy[i]<data_copy[i+1])):
            di_x.append(data_copy[i])
            di_weizhi.append(i)
            low_k += 1
            continue

    #2提取转折点
    number1,number2=0,0
    gaozhuan_x,gaozhuan_weizhi=[],[]
    dizhuan_x,dizhuan_weizhi=[],[]
    gaozhuan_x.append(data_copy[0])
    gaozhuan_weizhi.append(0)
    number1 += 1
    gaozhuan_x.append(data_copy[len(data) - 1])
    gaozhuan_weizhi.append(len(data) - 1)
    number1 += 1
    for i in range(2,high_k-1):
        if((gao_x[i]>=gao_x[i-1]) and (gao_x[i]>gao_x[i+1])):
            gaozhuan_x.append(gao_x[i])
            gaozhuan_weizhi.append(gao_weizhi[i])
            number1+=1
    for i in range(1,low_k-1):
        if((di_x[i]<=di_x[i-1]) and (di_x[i]<di_x[i+1])):
            dizhuan_x.append(di_x[i])
            dizhuan_weizhi.append(di_weizhi[i])
            number2+=1

    #3高低转折点排序，得转折点集
    PLR_data,PLR_number=[],[]
    for i in range(0,number1-1):
        PLR_number.append(gaozhuan_weizhi[i])
    for i in range(0,number2-1):
        PLR_number.append(dizhuan_weizhi[i])
    PLR_number.sort()
    for i in PLR_number:
        PLR_data.append(data[i])
    #plt.plot(data,'r')
    #plt.plot(PLR_number,PLR_data,'b')
    #plt.show()
    #print(PLR_number)
    #print(PLR_data)
    return PLR_number,PLR_data

k_distance=[]
def PLR_dis(train,test):
    train_t_PLR,train_x_PLR=PLR_WFTP(train)
    test_t_PLR,test_x_PLR=PLR_WFTP(test)
    #把时间合并
    union_time=[]
    for i in train_t_PLR:
        union_time.append(i)
    for i in test_t_PLR:
        union_time.append(i)
    x=set(union_time)
    union_time=list(x)
    union_time.sort()
    train_k,test_k=[],[]
    #计算斜率和斜率距离
    k_dis=0.0
    for i in range(1,len(union_time)):
        train_k.append((train[union_time[i]]-train[union_time[i-1]])*1.0/(union_time[i]-union_time[i-1]))
        test_k.append((test[union_time[i]]-test[union_time[i-1]])*1.0/(union_time[i]-union_time[i-1]))
    for i in range(1,len(union_time)):
        k_dis+=abs((union_time[i]-union_time[i-1])*1.0*(test_k[i-1]-train_k[i-1])/union_time[-1])
    k_distance.append(k_dis)
    return k_dis



#引入方差的SAX
paa_std=[]
paa_train_std,paa_test_std=[],[]
STD=0.0
def normalize(data):  # 正则化
    X = np.asanyarray(data)
    return (X - np.nanmean(X)) / np.nanstd(X)
def paa_trans(data):#分段
    data_copy=data.copy()
    data_copy = normalize(data)  # 类内函数调用：法1：加self：self.normalize()   法2：加类名：SAX_trans.normalize(self)
    paa_ts = []
    paa_std=[]
    n = len(data)
    xk = math.ceil(n / w)  # math.ceil()上取整，int()下取整
    for i in range(0, n, xk):
        #if (i+xk>=n):
            #temp_ts=data_copy[i:n-1]
        #else:
        temp_ts = data_copy[i:i + xk]
        paa_ts.append(np.mean(temp_ts))
        paa_std.append(np.std(temp_ts))#计算方差
        i += xk
    return paa_ts,paa_std
def to_sax(train,test):  # 转换成sax的字符串表示
    paa_train,paa_train_std = paa_trans(train)
    paa_test,paa_test_std=paa_trans(test)
    len_train = len(paa_train)
    len_test=len(paa_test)
    len_beta = len(beta)
    str_train,str_test = '',''
    for i in range(len_train):
        letter_found = False
        for j in range(len_beta):
            if np.isnan(paa_train[i]):
                str_train += '-'
                letter_found = True
                break
            if paa_train[i] < beta[j]:
                str_train += chr(aOffset + j)
                letter_found = True
                break
        if not letter_found:
            str_train += chr(aOffset + len_beta)
    for i in range(len_test):
        letter_found = False
        for j in range(len_beta):
            if np.isnan(paa_test[i]):
                str_test += '-'
                letter_found = True
                break
            if paa_test[i] < beta[j]:
                str_test += chr(aOffset + j)
                letter_found = True
                break
        if not letter_found:
            str_test += chr(aOffset + len_beta)
    #print(str_train,str_test)
    STD = 0.0
    for i in range(len(paa_test_std)):
        STD += (paa_test_std[i] - paa_train_std[i]) ** 2 * 1.0
    return str_train,str_test,STD
def compare_Dict():  # 生成距离表
    num_rep = range(alpha)  # 存放下标
    letters = [chr(x + aOffset) for x in num_rep]  # 根据alpha，确定字母的范围
    compareDict = {}
    len_letters = len(letters)
    for i in range(len_letters):
        for j in range(len_letters):
            if np.abs(num_rep[i] - num_rep[j]) <= 1:
                compareDict[letters[i] + letters[j]] = 0
            else:
                high_num = np.max([num_rep[i], num_rep[j]]) - 1
                low_num = np.min([num_rep[i], num_rep[j]])
                compareDict[letters[i] + letters[j]] = beta[high_num] - beta[low_num]
    return compareDict
def dist(train, test):  # 求出两个字符串之间的mindist()距离值
    strx1,strx2,STD=to_sax(train,test)
    len_strx1 = len(strx1)
    len_strx2 = len(strx2)
    com_dict = compare_Dict()
    if len_strx1 != len_strx2:
        print("The length of the two strings does not match")
    else:
        list_letter_strx1 = [x for x in strx1]
        list_letter_strx2 = [x for x in strx2]
        mindist = 0.0
        for i in range(len_strx1):
            if list_letter_strx1[i] is not '-' and list_letter_strx2[i] is not '-':
                mindist += (com_dict[list_letter_strx1[i] + list_letter_strx2[i]]) ** 2
        mindist = np.sqrt((len(train) * 1.0) / (w * 1.0)) * np.sqrt(mindist+STD*1.0)
        return mindist

SAX_dit=[]
def SAX(train,test):
    mindist = dist(train,test)
    SAX_dit.append(mindist)
    return mindist

PLR_STD_K_dis=[]
def PLR_STD_K(x):
    minn=9999999
    num=-1
    for i in range(len(train_data)):
        dis=SAX(train_data[i],test_data[x])+PLR_dis(train_data[i],test_data[x])*10
        if(dis<minn):
            minn,num=dis,i
        PLR_STD_K_dis.append(dis)
    return num



def one_NN():
    #print('SAX距离结果：')
    count=0
    accuracy=0
    for i in range(len(test_data)):
        a=PLR_STD_K(i)
        if train_label[a]==test_label[i]:
            print('预测标签：', train_label[a], '  ', '实际标签：', test_label[i],'  正确')
            count+=1
        else:
            print('预测标签：', train_label[a], '  ', '实际标签：', test_label[i],'  错误')
    accuracy=count*1.0/len(test_data)
    print('准确率为：',accuracy)

one_NN()