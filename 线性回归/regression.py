#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib2
import random

def loadDataSet(fileName):
    '''
    :param fileName:
    :return:
    '''
    numFeat = len(open(fileName).readline().strip().split('\t')) - 1 # 特征数目
    dataMat = [] # 数据矩阵
    labelMat = [] # 目标值矩阵
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        # 每行最后一列为目标值, 其它列为特征
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '''
    线性回归(Linear Regression): w = (X.T*X).I * (X.T*y)
    计算最佳拟合直线
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = np.mat(xArr) # 矩阵
    yMat = np.mat(yArr).T # 列向量
    xTx = xMat.T * xMat
    # np.linalg是NumPy提供的一个线性代数库
    # 矩阵求行列式, 行列式等于0, 矩阵不可逆
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    # 矩阵.I: 求矩阵的逆矩阵
    ws = xTx.I * (xMat.T * yMat) # w = (X.T*X).I * (X.T*y)
    # NumPy线性代数库提供了一个函数用来求解未知矩阵
    # ws = np.linalg.solve(xTx, xMat.T * yMat)

    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0], s=2, c='r')
    # 最佳拟合直线 y = Xw = w0*x0 + w1*x1
    # 首先需要对xMat进行排序, 然后再计算yHat
    xCopy = xMat.copy()
    xCopy.sort(0)
    ax.plot(xCopy[:, 1], xCopy * ws)
    plt.show()

    print np.corrcoef((xMat * ws).T, yMat.T) # 计算相关系数, 输入为行向量

    return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归(Locally Weighted Linear Regression): w = (X.T*W*X).I * (X.T*(W*y))
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    np.set_printoptions(linewidth=300,edgeitems=10)
    xMat = np.mat(xArr) # 矩阵
    yMat = np.mat(yArr).T # 列向量
    m = np.shape(xMat)[0]
    # numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
    # 关注第一个和第三个参数就行了.
    # 第一个参数: 输出方阵(行数=列数)的规模, 即行数或列数
    # 第三个参数: 默认情况下输出的是对角线全"1", 其余全"0"的方阵, 如果k为正整数, 则在右上方第k条对角线全"1"其余全"0',
    # k为负整数则在左下方第k条对角线全"1"其余全"0".
    # np.eye(2)
    # array([[1., 0.],
    #        [0., 1.]])
    # np.eye(3, k=1)
    # array([[0., 1., 0.],
    #        [0., 0., 1.],
    #        [0., 0., 0.]])
    weights = np.mat(np.eye((m))) # 对角权重矩阵, 权重矩阵是一个方阵, 阶数等于样本点个数, 即该矩阵为每个样本点初始化了一个权重
    # print 'weights = \n', weights
    # 遍历数据集, 计算每个样本点对应的权重值: 随着样本点与待预测点距离的递增, 权重将以指数级衰减
    for j in range(m):
        diffMat = testPoint - xMat[j, :] # 计算待预测点与样本点的差值(矩阵相减等于矩阵对应的各个元素相减)
        # diffMat * diffMat.T: 行向量*行向量的转置得到的列向量等于一个数值
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2)) # 计算权重, 参数k控制衰减的速度.
    # print 'weights = \n', weights
    # weights: m行xm列的方阵
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    # 矩阵.I: 求矩阵的逆矩阵
    ws = xTx.I * (xMat.T * (weights * yMat)) # w = (X.T*W*X).I * (X.T*(W*y))
    # NumPy线性代数库提供了一个函数用来求解未知矩阵
    # ws = np.linalg.solve(xTx, xMat.T * (weights * yMat))
    return ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    测试局部加权线性回归
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    m, n = np.shape(testArr) # 测试数据的行数和列数
    yHat = np.zeros(m) # 预测值
    wsMat = np.zeros((m, n)) # 测试数据的权重矩阵, 每一行为一个权重向量的转置
    for i in range(m): # 遍历测试数据集
        ws = lwlr(testArr[i], xArr, yArr, k) # 计算权重向量
        wsMat[i, :] = ws.T
        yHat[i] = testArr[i] * ws # 计算预测值

    xMat = np.mat(xArr) # 矩阵
    yMat = np.mat(yArr).T # 列向量
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(211)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0], s=2, c='r')
    # 最佳拟合直线 y = Xw = w0*x0 + w1*x1
    srtInd = xMat[:, 1].argsort(0) # 按照矩阵第2列从小到大进行排序, 返回排序后元素的索引值
    xSort = xMat[srtInd][:, 0, :]
    # print 'xSort = \n', xSort
    ax.plot(xSort[:, 1], yHat[srtInd])
    plt.xlabel('')
    plt.ylabel('')

    ax = fig.add_subplot(212)
    # ax.plot(wsMat[:, 1])
    ax.plot(wsMat)
    plt.xlabel(u'样本')
    plt.ylabel(u'样本权重')
    plt.show()

    print np.corrcoef(yHat.T, yMat.T) # 计算相关系数, 输入为行向量

    return yHat

def rssError(yArr, yHatArr):
    '''
    残差平方和RSS(Residual Sum of Squares)
    也称为误差平方和SSE(Sum of Squares for Error)
    :param yArr: 真实值
    :param yHatArr: 预测值
    :return:
    '''
    return ((yArr-yHatArr)**2).sum() # 返回误差平方和

def tssError(yArr, yMean):
    '''
    总平方和TSS(Total Sum of Squares)
    :param yArr: 真实值
    :param yMean: 真实值的平均值
    :return:
    '''
    return ((yArr-yMean)**2).sum() # 返回总平方和

def essError(yHatArr, yMean):
    '''
    解释平方和ESS(Explained Sum of Squares)
    也称为回归平方和SSR(Sum of Squares for Regression)
    :param yHatArr: 预测值
    :param yMean: 真实值的平均值
    :return:
    '''
    return ((yHatArr-yMean)**2).sum() # 返回回归平方和

def abaloneLwlrTest():
    '''
    在鲍鱼数据集上测试线性回归和局部加权线性回归算法
    :return:
    '''
    abX, abY = loadDataSet('abalone.txt')
    print '###############局部加权线性回归###############'
    yMean99 = np.mean(abY[0:99], 0)
    print '##########训练集##########'
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    rssErr01 = rssError(abY[0:99], yHat01.T)
    tssErr = tssError(abY[0:99], yMean99)
    essErr01 = essError(yHat01.T, yMean99)
    print 'rssErr01 = ', rssErr01
    print 'essErr01 = ', essErr01
    print 'tssErr = ', tssErr
    print 'essErr01 + rssErr01 = ', (essErr01 + rssErr01)
    print 'R方(k=0.1) = ', (1 - rssErr01 / tssErr)

    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    rssErr1 = rssError(abY[0:99], yHat1.T)
    tssErr = tssError(abY[0:99], yMean99)
    essErr1 = essError(yHat1.T, yMean99)
    print 'rssErr1 = ', rssErr1
    print 'essErr1 = ', essErr1
    print 'tssErr = ', tssErr
    print 'essErr1 + rssErr1 = ', (essErr1 + rssErr1)
    print 'R方(k=1) = ', (1 - rssErr1 / tssErr)

    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    rssErr10 = rssError(abY[0:99], yHat10.T)
    tssErr = tssError(abY[0:99], yMean99)
    essErr10 = essError(yHat10.T, yMean99)
    print 'rssErr10 = ', rssErr10
    print 'essErr10 = ', essErr10
    print 'tssErr = ', tssErr
    print 'essErr10 + rssErr10 = ', (essErr10 + rssErr10)
    print 'R方(k=10) = ', (1 - rssErr10 / tssErr)

    print '##########测试集##########'
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    rssErr01 = rssError(abY[100:199], yHat01.T)
    tssErr = tssError(abY[100:199], yMean99)
    essErr01 = essError(yHat01.T, yMean99)
    print 'rssErr01 = ', rssErr01
    print 'essErr01 = ', essErr01
    print 'tssErr = ', tssErr
    print 'essErr01 + rssErr01 = ', (essErr01 + rssErr01)
    print 'R方(k=0.1) = ', (1 - rssErr01 / tssErr)

    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    rssErr1 = rssError(abY[100:199], yHat1.T)
    tssErr = tssError(abY[100:199], yMean99)
    essErr1 = essError(yHat1.T, yMean99)
    print 'rssErr1 = ', rssErr1
    print 'essErr1 = ', essErr1
    print 'tssErr = ', tssErr
    print 'essErr1 + rssErr1 = ', (essErr1 + rssErr1)
    print 'R方(k=1) = ', (1 - rssErr1 / tssErr)

    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    rssErr10 = rssError(abY[100:199], yHat10.T)
    tssErr = tssError(abY[100:199], yMean99)
    essErr10 = essError(yHat10.T, yMean99)
    print 'rssErr10 = ', rssErr10
    print 'essErr10 = ', essErr10
    print 'tssErr = ', tssErr
    print 'essErr10 + rssErr10 = ', (essErr10 + rssErr10)
    print 'R方(k=10) = ', (1 - rssErr10 / tssErr)

    print '###############线性回归###############'
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    rssErr = rssError(abY[100:199], yHat.T.A)
    tssErr = tssError(abY[100:199], yMean99)
    essErr = essError(yHat.T.A, yMean99)
    print 'rssErr = ', rssErr
    print 'essErr = ', essErr
    print 'tssErr = ', tssErr
    print 'essErr + rssErr = ', (essErr + rssErr)
    print 'R方 = ', (1 - rssErr / tssErr)

def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    岭回归(Ridge Regression): w = (X.T*X + lambda * I).I * (X.T*y)
    :param xMat:
    :param yMat:
    :param lam: lambda
    :return: 回归系数
    '''
    xTx = xMat.T * xMat
    # 单位阵的行、列数等于数据集的特征数量
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    # 矩阵求行列式, 行列式等于0, 矩阵不可逆
    if np.linalg.det(denom) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    # 矩阵.I: 求矩阵的逆矩阵
    ws = denom.I * (xMat.T * yMat) # w = (X.T*X + lambda * I).I * (X.T*y)
    # NumPy线性代数库提供了一个函数用来求解未知矩阵
    # ws = np.linalg.solve(denom, xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    '''
    测试岭回归
    :param xArr:
    :param yArr:
    :return:
    '''
    # 数据标准化: 所有特征都减去各自的均值并除以方差
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yArr, 0) # 按列求y的均值
    yMat = yMat - yMean # y的各元素减去其均值
    xMeans = np.mean(xMat, 0) # 按列求x各列的均值
    xVar = np.var(xMat, 0) # 按列求x各列的方差
    xMat = (xMat - xMeans) / xVar # x的各列元素减去其各列元素的均值之后再除以各列的方差
    numTestPts = 30 # 不同lambda的数量
    wMat = np.zeros((numTestPts, np.shape(xMat)[1])) # 回归系数矩阵, 行为lambda的数量, 列为数据集的特征数量
    loglams = []
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10)) # 使用岭回归算法计算回归系数, 每次迭代使用不同的lambda
        wMat[i, :] = ws.T # 保存到回归系数矩阵
        loglams.append(i-10)
    return wMat, loglams

def abaloneRidgeTest():
    '''
    在鲍鱼数据集上测试岭回归算法
    :return:
    '''
    np.set_printoptions(linewidth=600)
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights, loglams = ridgeTest(abX, abY) # 使用岭回归算法计算回归系数
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    labels = [u'特征1', u'特征2', u'特征3', u'特征4', u'特征5', u'特征6', u'特征7', u'特征8']
    for i in range(np.shape(abX)[1]):
        ax.plot(loglams, ridgeWeights[:, i], label=labels[i])
        print ridgeWeights[:, i].T
    plt.legend(loc='upper right')
    plt.xlabel('log(lambda)')
    plt.ylabel(u'回归系数')
    plt.show()

def regularize(xMat):
    '''
    按照均值为0方差为1进行标准化处理
    :param xMat:
    :return:
    '''
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0) # 按列求均值
    inVar = np.var(inMat, 0) # 按列求方差
    inMat = (inMat - inMeans) / inVar # 各列元素减去其各列元素的均值然后再除以各列的方差
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    前向逐步线性回归
    :param xArr:
    :param yArr:
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0) # 按列求y的均值
    yMat = yMat - yMean # y的各元素减去其均值
    xMat = regularize(xMat) # 对x进行标准化处理
    m, n = np.shape(xMat)
    wMat = np.zeros((numIt, n)) # 回归系数矩阵, 行为迭代次数, 列为数据集的特征数量
    ws = np.zeros((n, 1)) # 回归系数, 初始值全为0
    # 为了实现贪心法而创建ws的两个副本
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = np.inf # 最小方差, 默认为正无穷大
        for j in range(n): # 遍历特征项
            for sign in [-1, 1]: # 分别计算减少和增加某个特征值后对误差的影响, 找到使得平方误差最小的回归系数
                wsTest = ws.copy() # 创建当前ws的副本
                wsTest[j] += eps * sign # sign为-1表示减少特征值, sign为1表示增加特征值
                yTest = xMat * wsTest # 计算预测值
                rssE = rssError(yMat.A, yTest.A) # 计算平方误差
                # 查找最小的平方误差并更新lowestError和wsMax
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest # 平方误差最小, 则ws最优
        ws = wsMax.copy() # 最优的回归系数
        wMat[i, :] = ws.T
    return wMat

def abaloneStageWiseTest():
    '''
    在鲍鱼数据集上测试前向逐步线性回归算法
    :return:
    '''
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = stageWise(abX, abY, eps=0.005, numIt=1000) # 使用前向逐步线性回归算法计算回归系数
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    labels = [u'特征1', u'特征2', u'特征3', u'特征4', u'特征5', u'特征6', u'特征7', u'特征8']
    for i in range(np.shape(abX)[1]):
        ax.plot(ridgeWeights[:, i], label=labels[i])
    plt.legend(loc='upper left')
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'回归系数')
    plt.show()

    xMat = np.mat(abX)
    yMat = np.mat(abY).T
    xMat = regularize(xMat)
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    ws = standRegres(xMat, yMat.T)
    print ws.T

def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    '''
    :param retX:
    :param retY:
    :param setNum:
    :param yr:
    :param numPce:
    :param origPrc:
    :return:
    '''
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5: # 过滤掉价格低于原始价格一半以上的套装
                    print "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print 'problem with item %d' % i

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def legoTest(lgX, lgY):
    '''
    :return:
    '''
    m, n = np.shape(lgX)
    lgX1 = np.mat(np.ones((m, n+1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = standRegres(lgX1, lgY)
    print ws.T

def crossValidation(xArr, yArr, numVal=10):
    '''
    交叉验证测试岭回归
    :param xArr:
    :param yArr:
    :param numVal: # 交叉验证次数
    :return:
    '''
    np.set_printoptions(linewidth=200)
    m = len(yArr) # 数据点个数
    indexList = range(m) # 索引列表
    errorMat = np.zeros((numVal, 30)) # 平方误差矩阵, 行数为交叉验证次数, 列数为30
    for i in range(numVal):
        trainX = [] # 训练集容器
        trainY = []
        testX = [] # 测试集容器
        testY = []
        random.shuffle(indexList) # 将序列indexList的所有元素随机排序
        # 分割数据集为训练集和测试集, 其中训练集占90%, 测试集占10%
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat, loglams = ridgeTest(trainX, trainY) # 使用岭回归算法计算回归系数
        for k in range(30):
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0) # 训练集均值
            varTrain = np.var(matTrainX, 0) # 训练集方差
            matTestX = (matTestX - meanTrain) / varTrain # 标准化测试集
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY) # 使用不同lambda值训练出来的回归系数计算预测值
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY)) # 计算平方误差
    meanErrors = np.mean(errorMat, 0) # 计算平方误差的均值
    minMean = float(min(meanErrors)) # # 获得平方误差的最小均值
    bestWeights = wMat[np.nonzero(meanErrors==minMean)] # 获得平方误差中均值最小的回归系数
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print 'the best model from Ridge Regression is:\n', unReg
    print 'with constant term: ', -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat)

    ws = standRegres(xArr, yArr)
    print ws.T

if __name__=='__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)
    # print ws

    # ws = lwlr(xArr[0], xArr, yArr, 0.1)
    # yHat = xArr[0] * ws
    # print 'Weights = ', ws.T, 'yHat = ', yHat, ', y = ', yArr[0]
    # ws = lwlr(xArr[0], xArr, yArr, 0.05)
    # yHat = xArr[0] * ws
    # print 'Weights = ', ws.T, 'yHat = ', yHat, ', y = ', yArr[0]

    # yHatArr = lwlrTest(xArr, xArr, yArr, 0.01)
    # print yHatArr

    # abaloneLwlrTest()
    # abaloneRidgeTest()
    # abaloneStageWiseTest()
    abX, abY = loadDataSet('abalone.txt')
    crossValidation(abX, abY)