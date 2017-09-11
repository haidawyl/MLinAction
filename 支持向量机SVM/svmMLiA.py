#!/usr/bin/python
#  -*- coding:utf-8 -*-

import math
import random
import numpy as np
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt

def loadDataSet(filename):
    '''
    读文件生成数据矩阵和类标签矩阵
    :param filename: 文件路径
    :return: 数据矩阵和类标签矩阵
    '''
    dataMat = [] # 数据矩阵
    labelMat = [] # 类标签矩阵
    fr = open(filename) # 打开文件
    for line in fr.readlines(): # 遍历所有行数据
        lineArr = line.strip().split('\t') # 截取掉每行的回车字符, 再使用tab字符 '\t' 将行数据分割成一个元素列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])]) # 数据矩阵加入第0列和第1列
        labelMat.append(float(lineArr[2])) # 类标签矩阵加入第2列
    return dataMat, labelMat # 返回数据矩阵和类标签矩阵

def selectJrand(i, m):
    '''
    返回一个[0, m]范围内且不同于i的整数
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    '''
    j = i
    while(j == i):
        j = int(random.uniform(0, m)) # 随机生成一个[0, m)范围内的实数, 并转换为整数
    return j

def clipAlpha(aj, H, L):
    '''
    调整大于H或小于L的alpha值, 即调整后的值位于[L, H]范围内
    :param aj: 待调整的值
    :param H: 右边界
    :param L: 左边界
    :return:
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版SMO算法
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :return: 常数b和向量alphas
    '''
    dataMatrix = np.mat(dataMatIn) # 将二维数组转换为NumPy矩阵
    labelMat = np.mat(classLabels).transpose() # 将一维数组转换为NumPy行向量, 然后再对行向量进行转置变成列向量
    b = 0
    m, n = np.shape(dataMatrix) # 取得矩阵的行数和列数
    alphas = np.mat(np.zeros((m, 1))) # 创建以0填充的m行x1列的alpha矩阵
    iter = 0 # 存储在没有任何alpha改变的情况下遍历数据集的次数
    # 在程序每次遍历数据集的过程中, 如果alpha值没有变化, 则iter+1.
    # 最后, 只有在所有数据集上遍历maxIter次, 且不再发生任何alpha修改之后, 程序才会停止并退出while循环.
    while(iter < maxIter): # 迭代次数不大于参数maxIter
        alphaPairsChanged = 0 # 记录本次迭代中alpha是否已经进行优化
        for i in range(m): # 遍历矩阵的每一行数据
            # np.multiply(alphas, labelMat).T: 两个矩阵对应元素相乘(即点乘), 然后再对矩阵进行转置(行变列, 列变行), 例:
            # alphas = [[1]
            #           [2]]
            # labelMat = [[3]
            #             [4]]
            # np.multiply(alphas, labelMat) = [[3]
            #                                  [8]]
            # np.multiply(alphas, labelMat).T = [[3 8]]

            # dataMatrix*dataMatrix[i, :].T: 矩阵乘以矩阵某一行的转置, 例
            # dataMatrix = [[1 2]
            #               [3 4]]
            # dataMatrix[0, :] = [[1 2]]
            # dataMatrix[0, :].T = [[1]
            #                       [2]]
            # dataMatrix*dataMatrix[0, :].T =[[1 2]  *  [[1]  =  [[1*1+2*2  =  [[5
            #                                 [3 4]]     [2]]      3*1+4*2]]     11]]

            # np.multiply(alphas, labelMat).T*dataMatrix*dataMatrix[0, :].T = [[3 8]] * [[5    = [[103]]
            #                                                                             11]]
            # f(x) = w.T * x + b
            # 计算预测的类别
            # 列向量alphas和列向量labelMat进行点乘得列向量后再进行转置得行向量,
            # 数据矩阵和数据矩阵某行向量转置后的列向量进行矩阵乘法得列向量,
            # 最后行向量和列向量进行矩阵乘法得1x1的矩阵, 即为一个数
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # 误差 = 预测结果 - 真实结果
            Ei = fXi - float(labelMat[i])
            # 如果误差很大, 则对该数据实例所对应的alpha值进行优化
            # 判断负间隔和正间隔,
            # 同时也要检查alpha值不能等于0或C, 由于后面alpha小于0或大于C时将被调整为0或C, 所以一旦在该if语句中
            # 它们等于这2个值的话, 那么它们就已经在"边界"上了, 因而不再能够减小或增大, 因此也就不值得再对它们进行优化了.
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m) # 随机选择第二个alpha值, 即alphas[j]
                # 针对第二个alpha值预测的类别
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                # 针对第二个alpha值的误差
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算L和H的值, 它们用于将alphas[j]调整到0到C之间.
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: # 如果L和H相等, 不需做任何改变
                    print 'L==H'
                    continue
                # eta是alphas[j]的最优修改量
                # eta = 2*X[i]*[j].T - X[i]*X[i].T - X[j]*X[j].T = -(X[i]-X[j])^2
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print 'eta>=0'
                    continue
                # 计算得到新的alphas[j]
                # Ei - Ej = (fXi - float(labelMat[i])) - (fXj - float(labelMat[j])) = fXi - fXj - float(labelMat[i]) + float(labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b - float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) - b - float(labelMat[i]) + float(labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) - float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) - float(labelMat[i]) + float(labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T) - np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + float(-labelMat[i] + labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T) - np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T) -labelMat[i] + labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T) - np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T) - labelMat[i] + labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*[dataMatrix*dataMatrix[i, :].T - dataMatrix*dataMatrix[j, :].T] - labelMat[i] + labelMat[j])
                # = float(np.multiply(alphas, labelMat).T*[dataMatrix*(dataMatrix[i, :].T - dataMatrix[j, :].T)] - labelMat[i] + labelMat[j])
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): # alphas[j]仅有轻微地改变
                    print 'j not moving enough'
                    continue
                # alphaJold - alphas[j]等于alphas[j]的改变量的负值
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j]) # alphas[i]和alphas[j]的改变方向相反
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                              dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                              dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print 'iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1 # 遍历完矩阵的所有数据以后, alpha值和b值没有变化, 继续对alpha值和b值进行优化, 如果连续迭代maxIter次以后alpha值和b值依然没有变化, 则结束优化过程, 我们可以认为当前的alpha值和b值是最优的.
        else: iter = 0 # 遍历完矩阵的所有数据以后, alpha值和b值有变化, 说明alpha值和b值还有优化的可能性, 那么需要对alpha值和b值继续进行优化, 迭代次数置为0.
        print 'iteration number: %d' % iter
    return b, alphas # 返回常数b和向量alphas

class optStruct:
    # def __init__(self, dataMatIn, classLabels, C, toler):
    #     '''
    #     :param dataMatIn: 数据集
    #     :param classLabels: 类别标签
    #     :param C: 常数C
    #     :param toler: 容错率
    #     '''
    #     self.X = dataMatIn
    #     self.labelMat = classLabels
    #     self.C = C
    #     self.tol = toler
    #     self.m = np.shape(dataMatIn)[0] # 取得数据集矩阵的行数
    #     self.alphas = np.mat(np.zeros((self.m, 1))) # 创建以0填充的m行x1列的alpha矩阵
    #     self.b = 0
    #     self.eCache = np.mat(np.zeros((self.m, 2))) # 创建用于缓存误差的m行x2列矩阵, 矩阵的第1列存储eCache是否有效的标志位, 第2列存储实际的误差值E

    def __init__(self, dataMatIn, classLabels, C, toler, kTup=('lin', 0)):
        '''
        :param dataMatIn: 数据集
        :param classLabels: 类别标签
        :param C: 常数C
        :param toler: 容错率
        :param kTup: 包含核函数信息的元祖
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]  # 取得数据集矩阵的行数
        self.alphas = np.mat(np.zeros((self.m, 1))) # 创建以0填充的m行x1列的alpha矩阵
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) # 创建用于缓存误差的m行x2列矩阵, 矩阵的第1列存储eCache是否有效的标志位, 第2列存储实际的误差值E
        self.K = np.mat(np.zeros((self.m, self.m))) # 创建以0填充的m行xm列的矩阵K
        for i in range(self.m): # 遍历矩阵K
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup) # 矩阵K的第i列进行核函数运算

def calcEk(oS, k):
    '''
    计算误差值E
    :param oS:
    :param k:
    :return:
    '''
    # 计算预测的类别
    # fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    # 为了使用核函数而进行的修改
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    # 误差 = 预测结果 - 真实结果
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    '''
    内循环中启发式方法, 用于选择第2个alpha值
    该函数的目标是选择合适的第2个alpha值以保证在每次优化中采用最大步长
    :param i:
    :param oS:
    :param Ei:
    :return:
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei] # 将输入值Ei在缓存中设置成为有效的, 有效(valid)意味着已经计算好了
    # np.nonzero() 返回包含以输入列表为目录的非0列表值构成的一个新列表
    # matrix.A: Return matrix self as an ndarray object.Equivalent to np.asarray(matrix).
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0] # 创建由非零E值所对应的alpha值的index(索引)构成的新列表
    if (len(validEcacheList)) > 1:
        # 遍历所有非零E值所对应的alpha值, 选择其中使得改变最大的值
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k) # 计算误差值Ek
            deltaE = abs(Ei - Ek) # 计算误差值改变量的绝对值
            if (deltaE > maxDeltaE): # 选择误差值改变量绝对值的最大值
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else: # 第一次循环时随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j) # 计算误差值Ej
    return j, Ej

def updateEk(oS, k):
    '''
    更新缓存中的误差值
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    '''
    优化alpha的内循环
    :param i:
    :param os:
    :return: 是否对alpha对进行了优化, 1为是, 0为否
    '''
    Ei = calcEk(oS, i) # 计算误差值Ei
    # 如果误差很大, 则对该数据实例所对应的alpha值进行优化
    # 判断负间隔和正间隔,
    # 同时也要检查alpha值不能等于0或C, 由于后面alpha小于0或大于C时将被调整为0或C, 所以一旦在该if语句中
    # 它们等于这2个值的话, 那么它们就已经在"边界"上了, 因而不再能够减小或增大, 因此也就不值得再对它们进行优化了.
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei) # 使用启发式方法选择第2个alpha值和alpha值的误差
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 计算L和H的值, 它们用于将alphas[j]调整到0到C之间.
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: # 如果L和H相等, 不需做任何改变
            print 'L==H'
            return 0
        # eta是alphas[j]的最优修改量
        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        # 为了使用核函数而进行的修改
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print 'eta>=0'
            return 0
        # 计算得到新的alphas[j]
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j) # 更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): # alphas[j]仅有轻微地改变
            print 'j not moving enough'
            return 0
        # alphaJold - alphas[j]等于alphas[j]的改变量的负值
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j]) # alphas[i]和alphas[j]的改变方向相反
        updateEk(oS, i) # 更新误差缓存
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * \
        #                  oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        # 为了使用核函数而进行的修改
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * \
        #                  oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 为了使用核函数而进行的修改
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    完整版SMO算法
    (1). 进行一次全数据集alpha值优化处理;
    (2). 循环进行非边界alpha值优化处理过程直至在处理过程中没有alpha值再变化;
    (3). 顺序执行(1)和(2) 直至遍历超过maxIter次, 或者遍历整个数据集都未对任意alpha对进行修改
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :param kTup: 包含核函数信息的元祖
    :return: 常数b和向量alphas
    '''
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0 # 存储在没有任何alpha改变的情况下遍历数据集的次数
    entireSet = True # 遍历全部数据集标识, 默认为True
    alphaPairsChanged = 0  # 记录本次迭代中alpha是否已经进行优化
    # 只有在所有数据集上遍历超过maxIter次, 或者遍历整个数据集都未对任意alpha对进行修改时, 程序才会停止并退出while循环.
    # (alphaPairsChanged > 0) or (entireSet) == False: 遍历整个数据集之后设置entireSet为False, 同时未对任意alpha对进行修改, 则alphaPairsChanged为0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)): # 迭代次数不大于参数maxIter, 并且对任意alpha对有修改时
        alphaPairsChanged = 0 # 记录本次迭代中alpha是否已经进行优化
        if entireSet: # 全数据集alpha值优化处理过程
            for i in range(oS.m):  # 遍历矩阵的每一行数据
                alphaPairsChanged += innerL(i, oS) # 优化alpha值
                print 'fullSet, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged)
            iter += 1
        else: # 非边界alpha值优化处理过程
            nonBoundIds = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] # 取得非边界alpha值的index(索引), 即不在边界0或C上的alpha值的index(索引)
            for i in nonBoundIds: # 遍历非边界alpha值
                alphaPairsChanged += innerL(i, oS) # 优化非边界的alpha值
                print 'non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False # 下次循环进入非边界alpha值优化处理过程
        elif (alphaPairsChanged == 0): # 在没有alpha值变化时
            entireSet = True # 下次循环进入全数据集alpha值优化处理过程
        print 'iteration number: %d' % iter
    return oS.b, oS.alphas # 返回常数b和向量alphas

def calcWs(alphas, dataArr, classLabels):
    '''
    基于样本数据集和在样本数据集上计算得到的alpha值, 计算w的加和
    如何对数据进行分类?数据行向量和计算得到的w加和生成的列向量进行矩阵乘法, 得到的数值再加上b,
    如果得到的数值是负数则属于-1类, 正数则属于+1类
    :param alphas:
    :param dataArr:
    :param classLabels:
    :return:
    '''
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1)) # 矩阵X的列数 = 列向量的行数
    for i in range(m): # 矩阵X的行数据
        # print 'alphas[', i, '] = ', alphas[i]
        # print 'labelMat[', i, '] = ', labelMat[i]
        # print 'X[', i, '].T = \n', X[i, :].T
        # 列向量alphas的i行(即一个数值)和列向量labelMat的i行(也是一个数值)相乘得到一个数值
        # 再和矩阵X的i行的转置得到的列向量进行点乘, 得到的结果累加到w上.
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
        # print 'w = \n', w
    return w # 返回n行(等于矩阵X的列数)x1列的矩阵

def testSMO():
    '''
    测试SMO
    :return:
    '''
    dataArr, labelArr = loadDataSet('testSet.txt')
    # print 'dataArr = \n', dataArr
    # print 'labelArr = \n', labelArr

    # sBeginTime = time()
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # sEndTime = time()

    pBeginTime = time()
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    pEndTime = time()

    print 'b = ', b
    print 'alphas.shape = ', np.shape(alphas)
    print 'alphas = ', alphas[alphas>0]

    # print 'Simple SMO time = ', (sEndTime - sBeginTime)
    print 'SMO time = ', (pEndTime - pBeginTime)

    ws = calcWs(alphas, dataArr, labelArr)
    print 'ws = ', ws

    minVal = 0
    maxVal = 10
    # 输出支持向量
    for i in range(len(alphas)):
        if alphas[i] > 0.0:
            predict = np.mat(dataArr[i]) * np.mat(ws) + b
            print dataArr[i],labelArr[i], predict
            if predict == abs(predict):
                if (dataArr[i][0] < maxVal):
                    maxVal = dataArr[i][0]
            else:
                if (dataArr[i][0] > minVal):
                    minVal = dataArr[i][0]
    print minVal, maxVal

    dataMat = np.mat(dataArr)
    m = np.shape(dataMat)[0]
    # 记录每个数据点的类别估计值的m行x1列的列向量, 初始值全部为0
    trainingClassEst = np.zeros((m, 1))
    for i in range(len(dataArr)):
        predict = dataMat[i] * np.mat(ws) + b # 大于0属于+1类; 小于0属于-1类
        trainingClassEst[i] = predict
        if predict / abs(predict) != labelArr[i]:
            print 'the real class is %d but the predicted class is %d' % (labelArr[i], predict)

    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    plt.scatter(np.array(dataArr)[np.array(labelArr) > 0][:, 0], np.array(dataArr)[np.array(labelArr) > 0][:, 1], marker='^', alpha=1)
    plt.scatter(np.array(dataArr)[np.array(labelArr) < 0][:, 0], np.array(dataArr)[np.array(labelArr) < 0][:, 1], marker='o', alpha=1)

    X1 = np.arange(3.0, 6.0, 0.1)
    # 令f(w.Tx+b)=0, 则x2=(-b-ws1*x1)/ws2
    X2 = (-b - ws[0][0] * X1) / ws[1][0]
    plt.plot(X1, np.array(X2)[0, :])

    plt.show()

    plotROC(np.mat(trainingClassEst).T, labelArr, u'训练集ROC曲线')

def kernelTrans(X, A, kTup):
    '''
    :param X: 所有数据集
    :param A: 数据集中的一行
    :param kTup: 包含核函数信息的元祖, 元祖的第一个参数是描述所用核函数类型的字符串, 其它参数则是核函数可能需要的可选参数.
    :return:
    '''
    m, n = np.shape(X) # 取得矩阵X的行数和列数
    K = np.mat(np.zeros((m, 1))) # 创建以0填充的m行x1列的矩阵K
    if kTup[0] == 'lin': # 线性核函数
        K = X * A.T # 内积计算, 矩阵和行向量的转置得到的列向量进行矩阵乘法得到列向量
    elif kTup[0] == 'rbf': # 径向基核函数
        for j in range(m): # 遍历数据集的每一行
            deltaRow = X[j, :] - A # 计算j行与行A的矩阵差得到行向量
            K[j] = deltaRow * deltaRow.T # 行向量与行向量的转置得到的列向量进行矩阵乘法得到1x1的矩阵, 即为一个数, 该数赋值给列向量K的j行
        K = np.exp(K / (-1*kTup[1]**2)) # 计算高斯函数的值, 除法符号意味着对矩阵元素展开计算
    else: raise NameError('Houston Wh Have a Problem -- That Kernel is not recognized') # 抛出异常
    return K

def testRbf(k1=1.3):
    '''
    利用核函数进行分类的径向基测试函数
    :param k1: 高斯径向基函数的sigma
    :return:
    '''
    dataArr, labelArr = loadDataSet('testSetRBF.txt') # 读取数据集和类标签
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) # 运行Platt SMO算法, 核函数为径向基函数
    dataMat = np.mat(dataArr) # 数据矩阵
    labelMat = np.mat(labelArr).transpose() # 类标签矩阵
    # matrix.A: Return matrix self as an ndarray object.Equivalent to np.asarray(matrix).
    svInd = np.nonzero(alphas.A>0)[0] # 支持向量下标
    print 'svInd = ', svInd
    sVs = dataMat[svInd] # 支持向量数据集
    labelSV = labelMat[svInd] # 支持向量类标签
    print 'there are %d Support Vectors' % np.shape(sVs)[0]
    m, n = np.shape(dataMat) # 取得矩阵的行数和列数
    errorCount = 0 # 预测错误数
    # 记录每个数据点的类别估计值的m行x1列的列向量, 初始值全部为0
    trainingClassEst = np.zeros((m, 1))
    for i in range(m):
        # 此处仅使用了支持向量数据集, 为什么? 主要是因为非支持向量的alpha为0,
        # 在运算np.multiply(labelMat, alphas)时对应的列向量元素值为0, 然后其与使用核函数
        # 转换后的数据进行矩阵乘法时, 对应的乘积项也为0, 对于最终的加和运算没有影响
        # 所以此处仅使用了支持向量进行运算.
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1)) # 使用核函数转换原始数据
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b # 计算数据的预测值
        trainingClassEst[i] = predict
        if np.sign(predict) != np.sign(labelArr[i]): # 预测值和实际值不同, 则错误数+1
            errorCount += 1
    print 'the training error rate is: %f' % (float(errorCount)/m)

    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 6)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(211) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.scatter(np.array(dataArr)[np.array(labelArr) > 0][:, 0], np.array(dataArr)[np.array(labelArr) > 0][:, 1], marker='o', alpha=0.5)
    plt.scatter(np.array(dataArr)[np.array(labelArr) < 0][:, 0], np.array(dataArr)[np.array(labelArr) < 0][:, 1], marker='s', alpha=0.5)
    plt.title(u'训练数据集')

    dataArr, labelArr = loadDataSet('testSetRBF2.txt') # 在测试集上执行上述过程
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    # 记录每个数据点的类别估计值的m行x1列的列向量, 初始值全部为0
    testClassEst = np.zeros((m, 1))
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        testClassEst[i] = predict
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print 'the test error rate is: %f' % (float(errorCount)/m)

    plt.subplot(212) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.scatter(np.array(dataArr)[np.array(labelArr) > 0][:, 0], np.array(dataArr)[np.array(labelArr) > 0][:, 1], marker='o', alpha=0.5)
    plt.scatter(np.array(dataArr)[np.array(labelArr) < 0][:, 0], np.array(dataArr)[np.array(labelArr) < 0][:, 1], marker='s', alpha=0.5)
    plt.title(u'测试数据集')
    plt.show()

    plotROC(np.mat(trainingClassEst).T, labelArr, u'训练集ROC曲线')
    plotROC(np.mat(testClassEst).T, labelArr, u'测试集ROC曲线')

def img2vector(filename):
    '''
    :param filename: 文件路径
    :return:
    '''
    returnVect = np.zeros((1, 1024)) # 创建以0填充的1行x1024列的矩阵
    fr = open(filename) # 打开文件
    # 遍历文件的前32行
    for i in range(32):
        lineStr = fr.readline() # 读取文件一行
        # 遍历当前行的前32列
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j]) # 将32x32的矩阵转换为1x1024的矩阵
    return returnVect # 返回图像数据矩阵

def loadImages(dirName):
    '''
    加载图像文件
    :param dirName: 图像目录
    :return:
    '''
    from os import listdir
    hwLabels = [] # 手写数字标签向量
    trainingFileList = listdir(dirName) # 获取数据集目录的文件列表
    m = len(trainingFileList) # 获取数据集的文件数量
    trainingMat = np.zeros((m, 1024)) # 创建以0填充的m行x1024列的矩阵
    # 遍历样本文件
    for i in range(m):
        fileNameStr = trainingFileList[i] # 取得文件名
        fileStr = fileNameStr.split('.')[0] # 截取掉文件名的(.扩展名)
        classNumStr = int(fileStr.split('_')[0]) # 取得文件表示的真实数字
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr)) # 读取文件内容并加入数据集
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    '''
    测试手写数字识别
    :param kTup: 包含核函数信息的元祖
    :return:
    '''
    dataArr, labelArr = loadImages('trainingDigits') # 获取训练数据集和类标签
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup) # 运行Platt SMO算法, 核函数为径向基函数
    dataMat = np.mat(dataArr) # 数据矩阵
    labelMat = np.mat(labelArr).transpose() # 类标签矩阵
    # matrix.A: Return matrix self as an ndarray object.Equivalent to np.asarray(matrix).
    svInd = np.nonzero(alphas.A>0)[0] # 支持向量下标
    print 'svInd = ', svInd
    sVs = dataMat[svInd] # 支持向量数据集
    labelSV = labelMat[svInd] # 支持向量类标签
    print 'there are %d Support Vectors' % np.shape(sVs)[0]
    m, n = np.shape(dataMat) # 取得矩阵的行数和列数
    errorCount = 0 # 预测错误数
    # 记录每个数据点的类别估计值的m行x1列的列向量, 初始值全部为0
    trainingClassEst = np.zeros((m, 1))
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup) # 使用核函数转换原始数据
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b # 计算数据的预测值
        trainingClassEst[i] = predict
        if np.sign(predict) != np.sign(labelArr[i]): # 预测值和实际值不同, 则错误数+1
            errorCount += 1
    print 'the training error rate is: %f' % (float(errorCount)/m)
    plotROC(np.mat(trainingClassEst).T, labelArr, u'训练集ROC曲线')

    dataArr, labelArr = loadImages('testDigits') # 获取测试数据集和类标签
    errorCount = 0 # 预测错误数
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    # 记录每个数据点的类别估计值的m行x1列的列向量, 初始值全部为0
    testClassEst = np.zeros((m, 1))
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup) # 使用核函数转换原始数据
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b # 计算数据的预测值
        testClassEst[i] = predict
        if np.sign(predict) != np.sign(labelArr[i]): # 预测值和实际值不同, 则错误数+1
            errorCount += 1
    print 'the training error rate is: %f' % (float(errorCount)/m)
    plotROC(np.mat(testClassEst).T, labelArr, u'测试集ROC曲线')

def plotROC(predStrengths, classLabels, title):
    '''
    <0.0, 0.0>: 将所有样例判为反例, 则TP=FP=0
    <1.0, 1.0>: 将所有样例判为正例, 则FN=TN=0
    x轴表示假阳率(FP/(FP+TN)), 在<0.0, 0.0>点假阳率等于0, 在<1.0, 1.0>点假阳率等于1
    y轴表示真阳率(TP/(TP+FN)), 在<0.0, 0.0>点真阳率等于0, 在<1.0, 1.0>点真阳率等于1
    :param predStrengths: 行向量, 表示分类结果的预测强度,
    如果值为负数则值越小被判为反例的预测强度越高, 反之值为正数则值越大被判为正例的预测强度越高
    :param classLabels: 类别标签
    :return:
    '''
    cur = (1.0, 1.0) # 绘制光标的位置, 起始点为右上角<1.0, 1.0>的位置
    ySum = 0.0 # 计算AUC(Area Under the Curve, ROC曲线下面的面积)的值
    numPosClas = sum(np.array(classLabels) == 1.0) # 真实正例的数目
    yStep = 1 / float(numPosClas) # y轴的步长
    # len(classLabels) - numPosClas: 真实反例的数目
    xStep = 1 / float(len(classLabels) - numPosClas) # x轴的步长
    # 分类器的预测强度从小到大排序的索引列表
    # 得到的是样本预测强度从小到大(从负数到正数)的样例的索引值列表
    # 第一个索引指向的样例被判为反例的强度最高
    # 最后一个索引指向的样例被判为正例的强度最高
    sortedIndicies = predStrengths.argsort()
    # print predStrengths[0, sortedIndicies]
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历分类器的预测强度从小到大排序的索引列表
    # 先从排名最低的样例开始, 所有排名更低的样例都被判为反例, 而所有排名更高的样例都被判为正例.
    # 第一个值对应点为<1.0, 1.0>, 而最后一个值对应点为<0.0, 0.0>.
    # 然后, 将其移到排名次低的样例中去, 如果该样例属于正例, 那么对真阳率进行修改;
    # 如果该样例属于反例, 那么对假阳率进行修改.

    # 初始时预测强度最小, 那么所有的样本都被判为正例, 即对应图中右上角的位置.
    # 向后迭代的过程中, 预测强度依次增大, 则排名低(列表前面)的样本被判为反例, 排名高(列表后面)的样本被判为正例.
    # 如果当前样本为真实正例, 在将其预测为反例时, 则为伪反例FN, 根据真阳率=TP/(TP+FN), 因此真阳率下降, 沿y轴下移
    # 如果当前样本为真实反例, 在将其预测为反例时, 则为真反例TN, 根据假阳率=FP/(FP+TN), 因此假阳率下降, 沿x轴左移
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0: # 标签为1.0的类即正例, 则要沿着y轴的方向下降一个步长, 也就是要不断降低真阳率
            delX = 0
            delY = yStep
        else: # 标签为其它(0或者-1)的类即反例, 则要沿着x轴的方向倒退一个步长, 也就是要不断降低假阳率
            delX = xStep
            delY = 0
            # 为了计算AUC, 需要对多个小矩形的面积进行累加. 这些小矩形的宽度是xStep, 因此可以先对所有矩形
            # 的高度进行累加, 最后再乘以xStep得到其总面积. 所有高度的和(ySum)随着x轴的每次移动而渐次增加
            ySum += cur[1]
        # 一旦决定了是在x轴还是y轴方向上进行移动, 就可以在当前点和新点之间画出一条线段
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        # 更新当前点cur
        cur = (cur[0]-delX, cur[1]-delY)
    # 画出左下角到右上角之间的虚线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel(u'假阳率(False Positive Rate)')
    plt.ylabel(u'真阳率(True Positive Rate)')
    plt.title(title)
    ax.axis([0, 1, 0, 1])
    plt.show()
    print 'the Area Under the Curve is: ', ySum*xStep

if __name__=='__main__':
    # testSMO()
    # testRbf()
    testDigits()
