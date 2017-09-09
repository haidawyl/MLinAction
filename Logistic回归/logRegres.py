#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

def loadDataSet():
    '''
    读取文件加载数据集
    :return:
    '''
    dataMat = [] # 数据矩阵
    labelMat = [] # 类别标签向量
    fr = open('testSet.txt') # 打开文件
    for line in fr.readlines(): # 遍历所有行数据
        lineArr = line.strip().split() # 截取掉每行的回车字符, 再使用空格字符 ' ' 将行数据分割成一个元素列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 数据矩阵加入1.0以及文件的第0列和第1列, 生成3维数据[x0, x1, x2]
        labelMat.append(int(lineArr[2])) # 类别标签矩阵加入第2列
    return dataMat, labelMat # 返回数据矩阵和类别标签矩阵

def sigmoid(inX):
    '''
    sigmoid函数, sigma(z) = 1 / (1 + np.exp(-z))
    :param inX:
    :return:
    '''
    # 如果inX是一个向量或数组, 则np.exp(-inX)是针对其中的每一个元素进行运算的, 得到的结果仍然是一个向量或数组
    return np.longfloat(1.0 / (1 + np.exp(-inX)))

def gradAscent(dataMatIn, classLabels):
    '''
    批量梯度上升算法
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :return: 回归系数
    '''
    dataMatrix = np.mat(dataMatIn) # 将数组转换为NumPy矩阵
    labelMat = np.mat(classLabels).transpose() # 将一维数组转换为NumPy行向量, 然后再对行向量进行转置变成列向量
    m, n = np.shape(dataMatrix) # 取得数据矩阵的行数和列数
    alpha = 0.001 # 向目标移动的步长
    maxCycles = 500 # 最大迭代次数
    weights = np.ones((n, 1)) # 回归系数, 创建以1填充的n行x1列的NumPy数组
    # 迭代maxCycles后得到回归系数
    for k in range(maxCycles):
        # dataMatrix: m x n
        # weights: n x 1
        # resultMatrix = dataMatrix * weights: m x 1
        # dataMatrix[0][0] * weights[0] + dataMatrix[0][1] * weights[1] + dataMatrix[0][2] * weights[2] + ... + dataMatrix[0][n] * weights[n] = resultMatrix[0][0]
        # dataMatrix[1][0] * weights[0] + dataMatrix[1][1] * weights[1] + dataMatrix[1][2] * weights[2] + ... + dataMatrix[1][n] * weights[n] = resultMatrix[1][0]
        # dataMatrix[2][0] * weights[0] + dataMatrix[2][1] * weights[1] + dataMatrix[2][2] * weights[2] + ... + dataMatrix[2][n] * weights[n] = resultMatrix[2][0]
        # ...
        # dataMatrix[m][0] * weights[0] + dataMatrix[m][1] * weights[1] + dataMatrix[m][2] * weights[2] + ... + dataMatrix[m][n] * weights[n] = resultMatrix[m][0]
        # 对列向量运行sigmoid函数, 返回的结果还是列向量
        # h: m x 1
        h = sigmoid(dataMatrix * weights) # 得到预测类别的值
        # 向量相减等于向量中对应的元素相减, 得到的结果还是向量
        # labelMat: m x 1
        # error: m x 1
        error = (labelMat - h) # 计算真实类别与预测类别的差值
        # dataMatrix; m x n
        # dataMatrix.transpose(): n x m
        # error: m x 1
        # dataMatrix.transpose() * error: n x 1 本次计算得到的梯度
        # 数字乘以向量等于该数字分别乘以向量中的每一个元素
        # alpha * dataMatrix.transpose() * error: n x 1
        # weights: n x 1
        # 向量相加等于向量中对应的元素相加, 得到的结果还是向量
        weights = weights + alpha * dataMatrix.transpose() * error # 按照真实类别与预测类别的差值方向调整回归系数
    return weights # 返回回归系数

def plotBestFit(weights):
    '''
    :param weights: 回归系数
    :return:
    '''
    dataMat, labelMat = loadDataSet() # 读取数据集和类别标签
    dataArr = np.array(dataMat) # 将数据集转换成NumPy数组
    n = np.shape(dataArr)[0] # 获取数据集的行数
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n): # 遍历数据集, 分开标签为0和1的数据
        if int(labelMat[i]) == 1: # 标签为1的数据
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else: # 标签为0的数据
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure() # 画图
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s= 30, c='r', marker='s') # 画标签为1的散点图
    ax.scatter(xcord2, ycord2, s= 30, c='g') # 画标签为0的散点图
    # 创建区间为[-0.3, 0.3), 步长为0.1的等差numpy.ndarray数组
    x1 = np.arange(-3.0, 3.0, 0.1) # 我们要画的横轴数据
    # 此处令sigmoid函数为0. 0是两个分类(即类别l和类别0)的分界处. 因此我们设定0 = w0x0 + w1x1 + w2x2,
    # 在生成数据时, 我们已将x0设置为1.0, 则有x2 = (-weights[0] - weights[1] * x1) / weights[2]
    x2 = (-weights[0] - weights[1] * x1) / weights[2] # 根据公式计算得出纵轴数据
    ax.plot(x1, x2) # 画出分割线
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels, numIter=10):
    '''
    随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 类别标签
    :return: 回归系数
    '''
    m, n = np.shape(dataMatrix) # 取得数据集的行数和列数
    alpha = 0.01 # 向目标移动的步长
    weights = np.ones(n) # 回归系数, 创建以1填充的长度为n的NumPy数组

    # 记录回归系数每次改变后的值, 只记录前3维特征的回归系数变化情况
    x0 = []
    x1 = []
    x2 = []
    for i in range(m * numIter): # 遍历numIter次数据集
        # dataMatrix[i]; 长度为n的NumPy数组
        # weights; 长度为n的NumPy数组
        # dataMatrix[i] * weights: 进行点乘, 得到长度为n的NumPy数组
        # np.sum(dataMatrix[i] * weights): 对NumPy数组所有元素加和得到一个数字
        # 调用sigmoid函数, 参数是一个数字, 得到的h也是一个数字
        h = sigmoid(np.sum(dataMatrix[i%m] * weights)) # 得到预测类别的值
        # classLabels: 长度为m的列表
        # classLabels[i]: 列表的一个元素, 因此是一个数字
        error = classLabels[i%m] - h # 计算真实类别与预测类别的差值
        # dataMatrix[i]; 长度为n的NumPy数组
        # 一个数字乘以一个NumPy数组等于该数字分别乘以数组中的每一个元素
        # error * dataMatrix[i%m]: 本次计算得到的梯度
        # weights: 长度为n的NumPy数组
        # NumPy数组加上NumPy数组等于数组中对应的元素相加, 得到的结果还是NumPy数组
        weights = weights + alpha * error * dataMatrix[i%m] # 按照真实类别与预测类别的差值方向调整回归系数

        x0.append([i, weights[0]])
        x1.append([i, weights[1]])
        x2.append([i, weights[2]])

    # 画出回归系数的变化情况
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(311) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(np.mat(x0)[:, 0], np.mat(x0)[:, 1], alpha=0.5)
    plt.ylabel('X0')
    plt.subplot(312)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(np.mat(x1)[:, 0], np.mat(x1)[:, 1], alpha=0.5)
    plt.ylabel('X1')
    plt.subplot(313)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(np.mat(x2)[:, 0], np.mat(x2)[:, 1], alpha=0.5)
    plt.ylabel('X2')
    plt.show()

    return weights # 返回回归系数

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    改进后的随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 类别标签
    :param numIter: 迭代次数
    :return: 回归系数
    '''
    m, n = np.shape(dataMatrix) # 取得数据集的行数和列数
    weights = np.ones(n) # 回归系数, 创建以1填充的长度为n的NumPy数组

    # 记录回归系数每次改变后的值, 只记录前3维特征的回归系数变化情况
    x0 = []
    x1 = []
    x2 = []
    for j in range(numIter): # 遍历numIter次
        dataIndex = range(m) # 生成[0, m)的整数列表, 即为数据集索引值的列表
        for i in range(m): # 遍历整个数据集
            # 虽然alpha在每次迭代时都会减小, 但是永远也不会减小到0, 这是因为存在一个常数项.
            # 这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响. 如果要处理的问题
            # 是动态变化的, 那么可以适当加大该常数项, 来确保新的值获得更大的回归系数.
            # 另外值得注意的一点是, 在降低alpha的函数中, alpha每次减少1/(j+i), 其中j是迭代次数,
            # i是样本点的下标. 这样当j<<max(i)时, alpha就不是严格下降的.
            alpha = 4 / (1.0 + j + i) + 0.01 # 每次动态减小alpha值
            # 通过随机选取样本来更新回归系数. 这种方法将减少周期性的波动.
            randIndex = int(random.uniform(0, len(dataIndex))) # 随机生成一个[0, m)范围内的实数, 并转换为整数, 即得到随机选取样本的索引值

            # dataMatrix[randIndex]; 长度为n的NumPy数组
            # weights; 长度为n的NumPy数组
            # dataMatrix[randIndex] * weights: 进行点乘, 得到长度为n的NumPy数组
            # np.sum(dataMatrix[randIndex] * weights): 对NumPy数组所有元素加和得到一个数字
            # 调用sigmoid函数, 参数是一个数字, 得到的h也是一个数字
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights)) # 得到预测类别的值
            # classLabels: 长度为m的列表
            # classLabels[randIndex]: 列表的一个元素, 因此是一个数字
            error = classLabels[randIndex] - h # 计算真实类别与预测类别的差值
            # dataMatrix[randIndex]; 长度为n的NumPy数组
            # 一个数字乘以一个NumPy数组等于该数字分别乘以数组中的每一个元素
            # error * dataMatrix[i%m]: 本次计算得到的梯度
            # weights: 长度为n的NumPy数组
            # NumPy数组加上NumPy数组等于数组中对应的元素相加, 得到的结果还是NumPy数组
            weights = weights + alpha * error * dataMatrix[randIndex] # 按照真实类别与预测类别的差值方向调整回归系数
            del(dataIndex[randIndex]) # 删除dataIndex列表中已经使用过的元素, 下次计算梯度时不再使用该样本, 保证所有的样本数据都能够参与计算梯度

            x0.append([j*m+i, weights[0]])
            x1.append([j*m+i, weights[1]])
            x2.append([j*m+i, weights[2]])

    # 画出回归系数的变化情况
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(311) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(np.mat(x0)[:, 0], np.mat(x0)[:, 1], alpha=0.5)
    plt.ylabel('X0')
    plt.subplot(312)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(np.mat(x1)[:, 0], np.mat(x1)[:, 1], alpha=0.5)
    plt.ylabel('X1')
    plt.subplot(313)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(np.mat(x2)[:, 0], np.mat(x2)[:, 1], alpha=0.5)
    plt.ylabel('X2')
    plt.show()

    return weights # 返回回归系数

def testGradAscent():
    '''
    测试批量梯度上升算法
    :return:
    '''
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights.getA())
    print weights

def testStocGradAscent0():
    '''
    测试随机梯度上升算法
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent0(np.array(dataArr), labelMat, numIter=200)
    plotBestFit(weights)
    print weights

def testStocGradAscent1():
    '''
    测试随机梯度上升算法
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataArr), labelMat, numIter=30)
    plotBestFit(weights)
    print weights

def classifyVector(inX, weights):
    '''
    执行分类操作
    :param inX: 特征向量
    :param weights: 回归系数
    :return:
    '''
    # 特征向量点乘回归系数, 再对乘积进行加和运算, 最后使用得到的和值调用Sigmoid算法得到预测值
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5: # 预测值大于0.5, 归入类别1
        return 1.0
    else: # 否则归入类别0
        return 0.0

def colicTest():
    '''
    :return:
    '''
    frTrain = open('horseColicTraining.txt') # 打开训练集文件
    frTest = open('horseColicTest.txt') # 打开测试集文件
    trainingSet = [] # 训练集
    trainingLabels = [] # 训练集类别标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t') # 截取掉每行的回车字符, 再使用tab字符 '\t' 将行数据分割成一个元素列表
        lineArr = []
        for i in range(21): # 前21列是特征数据
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) # 最后一列是类别标签
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels, 500) # 使用训练集计算回归系数向量
    errorCount = 0 # 预测错误数
    numTestVec = 0.0 # 测试样本数量
    # 导入测试集
    for line in frTest.readlines():
        numTestVec += 1.0 # 测试样本数量+1
        currLine = line.strip().split('\t') # 截取掉每行的回车字符, 再使用tab字符 '\t' 将行数据分割成一个元素列表
        lineArr = []
        for i in range(21): # 前21列是特征数据
            lineArr.append(float(currLine[i]))
        # 预测分类, 与实际分类不符, 错误总数+1
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    # 计算错误率
    errorRate = (float(errorCount) / numTestVec)
    print 'the error rate of this test is : %f' % errorRate
    return errorRate

def multiTest():
    '''
    执行10次训练和测试, 计算平均错误率
    :return:
    '''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests))

def drawSigmoid():
    '''
    :return:
    '''
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(211) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    x1 = np.arange(-5.0, 6.0, 0.1)
    y1 = sigmoid(x1)
    plt.plot(x1, y1)

    plt.subplot(212)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    x2 = np.arange(-60, 60, 1)
    y2 = sigmoid(x2)
    plt.plot(x2, y2)

    plt.show()

if __name__=='__main__':
    # testGradAscent()
    # testStocGradAscent0()
    # testStocGradAscent1()
    multiTest()
    # drawSigmoid()