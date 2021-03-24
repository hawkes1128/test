import numpy as np
import math
import matplotlib.pyplot as plt
train_path = ['D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Adiac\\Adiac_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\AllGestureWiimoteX\\AllGestureWiimoteX_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\AllGestureWiimoteY\\AllGestureWiimoteY_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\AllGestureWiimoteZ\\AllGestureWiimoteZ_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ArrowHead\\ArrowHead_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Beef\\Beef_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\BeetleFly\\BeetleFly_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\BirdChicken\\BirdChicken_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\BME\\BME_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Car\\Car_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CBF\\CBF_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Chinatown\\Chinatown_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ChlorineConcentration\\ChlorineConcentration_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CinCECGTorso\\CinCECGTorso_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Coffee\\Coffee_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Computers\\Computers_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CricketX\\CricketX_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CricketY\\CricketY_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CricketZ\\CricketZ_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Crop\\Crop_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DiatomSizeReduction\\DiatomSizeReduction_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DistalPhalanxOutlineAgeGroup\\DistalPhalanxOutlineAgeGroup_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DistalPhalanxOutlineCorrect\\DistalPhalanxOutlineCorrect_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DistalPhalanxTW\\DistalPhalanxTW_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DodgerLoopDay\\DodgerLoopDay_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DodgerLoopGame\\DodgerLoopGame_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DodgerLoopWeekend\\DodgerLoopWeekend_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Earthquakes\\Earthquakes_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ECG200\\ECG200_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ECG5000\\ECG5000_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ECGFiveDays\\ECGFiveDays_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ElectricDevices\\ElectricDevices_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\EOGVerticalSignal\\EOGVerticalSignal_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\EOGHorizontalSignal\\EOGHorizontalSignal_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\EthanolLevel\\EthanolLevel_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FaceAll\\FaceAll_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FaceFour\\FaceFour_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FacesUCR\\FacesUCR_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FiftyWords\\FiftyWords_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Fish\\Fish_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FordA\\FordA_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FordB\\FordB_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FreezerRegularTrain\\FreezerRegularTrain_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FreezerSmallTrain\\FreezerSmallTrain_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Fungi\\Fungi_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GestureMidAirD1\\GestureMidAirD1_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GestureMidAirD2\\GestureMidAirD2_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GestureMidAirD3\\GestureMidAirD3_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GesturePebbleZ1\\GesturePebbleZ1_TRAIN.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GesturePebbleZ2\\GesturePebbleZ2_TRAIN.tsv']#对于不同的数据集修改路径即可
test_path = ['D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ACSF1\\ACSF1_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Adiac\\Adiac_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\AllGestureWiimoteX\\AllGestureWiimoteX_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\AllGestureWiimoteY\\AllGestureWiimoteY_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\AllGestureWiimoteZ\\AllGestureWiimoteZ_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ArrowHead\\ArrowHead_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Beef\\Beef_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\BeetleFly\\BeetleFly_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\BirdChicken\\BirdChicken_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\BME\\BME_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Car\\Car_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CBF\\CBF_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Chinatown\\Chinatown_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ChlorineConcentration\\ChlorineConcentration_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CinCECGTorso\\CinCECGTorso_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Coffee\\Coffee_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Computers\\Computers_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CricketX\\CricketX_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CricketY\\CricketY_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\CricketZ\\CricketZ_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Crop\\Crop_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DiatomSizeReduction\\DiatomSizeReduction_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DistalPhalanxOutlineAgeGroup\\DistalPhalanxOutlineAgeGroup_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DistalPhalanxOutlineCorrect\\DistalPhalanxOutlineCorrect_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DistalPhalanxTW\\DistalPhalanxTW_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DodgerLoopDay\\DodgerLoopDay_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DodgerLoopGame\\DodgerLoopGame_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\DodgerLoopWeekend\\DodgerLoopWeekend_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Earthquakes\\Earthquakes_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ECG200\\ECG200_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ECG5000\\ECG5000_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ECGFiveDays\\ECGFiveDays_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\ElectricDevices\\ElectricDevices_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\EOGVerticalSignal\\EOGVerticalSignal_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\EOGHorizontalSignal\\EOGHorizontalSignal_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\EthanolLevel\\EthanolLevel_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FacesAll\\FacesAll_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FaceFour\\FaceFour_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FacesUCR\\FacesUCR_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FiftyWords\\FiftyWords_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Fish\\Fish_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FordA\\FordA_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FordB\\FordB_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FreezerRegularTrain\\FreezerRegularTrain_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\FreezerSmallTrain\\FreezerSmallTrain_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\Fungi\\Fungi_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GestureMidAirD1\\GestureMidAirD1_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GestureMidAirD2\\GestureMidAirD2_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GestureMidAirD3\\Gesture',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GesturePebbleZ1\\GesturePebbleZ1_TEST.tsv',
'D:\\UCRArchive_2018(1)\\UCRArchive_2018\\GesturePebbleZ2\\GesturePebbleZ2_TEST.tsv']#对于不同的数据集修改路径即可
train_data = np.loadtxt(train_path[11], delimiter='\t')
#MidAirD3_TEST.tsv',
test_data = np.loadtxt(test_path[11], delimiter='\t')
train_label=train_data[:,0]
test_label=test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data=np.delete(test_data,0,axis=1)

class SAX_trans:

    def __init__(self, ts, w, alpha):
        self.ts = ts
        self.w = w
        self.alpha = alpha
        self.aOffset = ord('a')  # 字符的起始位置，从a开始
        self.breakpoints = {'3': [-0.43, 0.43],
                            '4': [-0.67, 0, 0.67],
                            '5': [-0.84, -0.25, 0.25, 0.84],
                            '6': [-0.97, -0.43, 0, 0.43, 0.97],
                            '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                            '9':[-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                            '10':[-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]
                            }
        self.beta = self.breakpoints[str(self.alpha)]

    def normalize(self):  # 正则化
        X = np.asanyarray(self.ts)
        return (X - np.nanmean(X)) / np.nanstd(X)

    def paa_trans(self):  # 转换成paa
        tsn=self.normalize()  # 类内函数调用：法1：加self：self.normalize()   法2：加类名：SAX_trans.normalize(self)
        paa_ts = []
        n = len(tsn)
        xk = math.ceil(n / self.w)  # math.ceil()上取整，int()下取整
        for i in range(0, n, xk):
            temp_ts = tsn[i:i + xk]
            paa_ts.append(np.mean(temp_ts))
            i = i + xk
        return paa_ts

    def to_sax(self):  # 转换成sax的字符串表示
        tsn = self.paa_trans()
        len_tsn = len(tsn)
        len_beta = len(self.beta)
        strx = ''
        for i in range(len_tsn):
            letter_found = False
            for j in range(len_beta):
                if np.isnan(tsn[i]):
                    strx += '-'
                    letter_found = True
                    break
                if tsn[i] < self.beta[j]:
                    strx += chr(self.aOffset + j)
                    letter_found = True
                    break
            if not letter_found:
                strx += chr(self.aOffset + len_beta)
        return strx

    def compare_Dict(self):  # 生成距离表
        num_rep = range(self.alpha)  # 存放下标
        letters = [chr(x + self.aOffset) for x in num_rep]  # 根据alpha，确定字母的范围
        compareDict = {}
        len_letters = len(letters)
        for i in range(len_letters):
            for j in range(len_letters):
                if np.abs(num_rep[i] - num_rep[j]) <= 1:
                    compareDict[letters[i] + letters[j]] = 0
                else:
                    high_num = np.max([num_rep[i], num_rep[j]]) - 1
                    low_num = np.min([num_rep[i], num_rep[j]])
                    compareDict[letters[i] + letters[j]] = self.beta[high_num] - self.beta[low_num]
        return compareDict

    def dist(self, strx1, strx2):  # 求出两个字符串之间的mindist()距离值
        len_strx1 = len(strx1)
        len_strx2 = len(strx2)
        com_dict = self.compare_Dict()

        if len_strx1 != len_strx2:
            print("The length of the two strings does not match")
        else:
            list_letter_strx1 = [x for x in strx1]
            list_letter_strx2 = [x for x in strx2]
            mindist = 0.0
            for i in range(len_strx1):
                if list_letter_strx1[i] is not '-' and list_letter_strx2[i] is not '-':
                    mindist += (com_dict[list_letter_strx1[i] + list_letter_strx2[i]]) ** 2
            mindist = np.sqrt((len(self.ts) * 1.0) / (self.w * 1.0)) * np.sqrt(mindist)
            return mindist


SAX_dit=[]
def SAX(x):
    SAX_dit.clear()
    for y in train_data:
        x1=SAX_trans(ts=test_data[x],w=80,alpha=10)
        x2=SAX_trans(ts=y,w=80,alpha=10)
        st1 = x1.to_sax()
        st2 = x2.to_sax()
        dist = x1.dist(st1, st2)
        SAX_dit.append(dist)
    return SAX_dit


def one_NN():
    #print('SAX距离结果：')
    count=0
    accuracy=0
    for i in range(len(test_data)):
        a=SAX(i)
        if train_label[a.index(min(a))]==test_label[i]:
            #print('预测标签：', train_label[a.index(min(a))], '  ', '实际标签：', test_label[i],'  正确')
            count+=1
        #else:
            #print('预测标签：', train_label[a.index(min(a))], '  ', '实际标签：', test_label[i],'  错误')
    accuracy=count*1.0/len(test_data)
    print('准确率为：',accuracy)

#plt.plot(test_data[4],'g')
#plt.plot(test_data[75],'y')
#plt.show()
#plt.plot(train_data[1],'r')
#plt.show()
'''
for i in range(10,len(train_path)):
    train_data = np.loadtxt(train_path[i], delimiter='\t')
    test_data = np.loadtxt(test_path[i], delimiter='\t')
    train_label = train_data[:, 0]
    test_label = test_data[:, 0]
    train_data = np.delete(train_data, 0, axis=1)
    test_data = np.delete(test_data, 0, axis=1)
    print(i+1,train_path[i])
    one_NN()
'''
x1 = SAX_trans(ts=test_data[1], w=80, alpha=10)
x2 = SAX_trans(ts=train_data[1], w=80, alpha=10)
st1 = x1.to_sax()
st2 = x2.to_sax()
dist = x1.dist(st1, st2)
print(dist)