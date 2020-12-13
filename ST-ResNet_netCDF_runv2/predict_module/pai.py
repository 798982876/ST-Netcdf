import numpy as np
from tqdm import tqdm

# 这段代码是用来计算pai
# pai 定义：(预测有案件的区域的实际案件总数/实际犯罪总数)/(预测犯罪的面积/研究区的面积)
# 步骤：
# 1 读取预测的结果 ，对测试集的预测结果和测试集的通过训练所得的模型进行预测，然后获取结果npy文件
# 2 针对全局，统计实际结果和预测结果全局的结果
# 3 获取预测犯罪的面积(统计其网格数)，统计不同网格分辨率的总体网格数
# 4 比值
# 5 考虑单个网格的对比
# 6 单个网格差异

# 累计百分比




# 计算面积 统计非0网格个数
def getCrimeArea(d):
    # 统计网格数边长
    length = len(d[0])
    gridCount = 0
    for i in range(length):
        for j in range(length):
            if d[i][j] != 0:
                gridCount = gridCount + 1
    return gridCount


# 计算预测位置犯罪数
def getPredCrimeNum(pred, real):
    length = len(pred[0])
    crimeCount = 0
    for i in range(length):
        for j in range(length):
            if pred[i][j] > 0:
                crimeCount = crimeCount + real[i][j]
    return crimeCount


# 计算犯罪总数
def getCrimeNum(d):
    length = len(d[0])
    crimeCount = 0
    for i in range(length):
        for j in range(length):
            crimeCount = crimeCount + d[i][j]
    return crimeCount


# 计算研究区域总面积
def getAllArea(d):
    length = len(d[0])
    allGrid = length * length
    return allGrid


# 计算pai
def getPai(predNum, predArea, realNum, Area):
    pai = 0.0
    if realNum != 0 and predArea != 0:
        pai = (predNum / realNum) / (predArea / Area)
    else:
        pai = 0.0
    return pai


# 计算pei
# 计算n* 就是统计有案件的网格数，找出其相同的由案件的网格数排名靠前的
# 首先得到crimeArea 一共有多少个网格有犯罪
# 对实际犯罪进行排序，取前多少个网格的犯罪
def getPei(predNum, predArea, real):
    n2 = []
    pei = []
    predAreaLength = len(predArea)
    # 对
    for i in range(predAreaLength):
        # 将实际的案件矩阵变为一维数组
        tempReal = real[i].tolist()
        tempReal2 = []
        for sublist in tempReal:
            for item in sublist:
                tempReal2.append(item)
        tempReal2 = sorted(tempReal2,reverse = True)
        # 对预测的网格数，计算排名前几的总数
        tempN=0
        for j in range(predArea[i]):
            if predArea[i] != 0:
                tempN = tempN + tempReal2[j]
            else:
                tempN = 0
        n2.append(tempN)
    tempPei = 0.0
    for i in range(len(n2)):
        if n2[i] != 0:
            tempPei = predNum[i] / n2[i]
        else:
            n2[i] = -1
        pei.append(tempPei)
    return pei,n2

def caculte_pai_pei(predict_path, real_path):
    a = np.load(predict_path)
    b = np.load(real_path)
    print(predict_path)
    # a = np.load("data/output/pred.npy")
    # b = np.load("data/output/real.npy")
    predLength = len(a)
    realLength = len(b)
    print('lena:',len(a))
    # 这四个参数分别是预测案件数、预测面积、真实案件数、研究面积
    predNum = []
    predArea = []
    realNum = []
    realArea = []
    pai = []
    Area = getAllArea(b[0])

    for i in range(predLength):
        predNum.append(getPredCrimeNum(a[i], b[i]))
        predArea.append(getCrimeArea(a[i]))
        realNum.append(getCrimeNum(b[i]))
    # 计算pai
    for i in range(len(predNum)):
        pai.append(getPai(predNum[i], realNum[i], predArea[i], Area))

    # 计算pei
    pei,n2 = getPei(predNum, predArea, b)
    for item in n2:
        if item == 0 :
            n2.remove(item)
    print(len(n2))
    print(sum(pei)/len(pei))

    for item in pai:
        if item == 0:
            pai.remove(item)

    print(sum(pai) / len(pai))
    return sum(pai) / len(pai)
