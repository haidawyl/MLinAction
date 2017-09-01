#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    k-近邻算法
    :param inX: 预测分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 训练样本集对应的标签向量, 元素数与dataSet的行数相同
    :param k: 选择最近邻居的数目
    :return: 发生频率最高的元素标签
    '''
    # dataSet.shape = (行数, 列数), 本例为(4, 2)
    dataSetSize = dataSet.shape[0]

    # 使用欧式距离公式计算2个向量点xA和xB之间的距离: d = ((xA0-xB0)**2 + (xA1-xB1)**2)**0.5
    # 例: 点(0,0)与(1,2)之间的距离计算为: d = ((1-0)**2 + (2-0)**2)**0.5
    # 点(1,0,0,1)与(7,6,9,4)之间的距离计算为: d = ((7-1)**2 + (6-0)**2 + (9-0)**2 + (4-1)**2)**0.5
    # np.tile(A,(m, n)), 将数组A重复m行, n列
    # 例: inX = [0, 0], np.tile(inX, (2, 1))的结果为
    # [[0 0]
    #  [0 0]]
    # np.tile(inX, (2, 2))的结果为
    # [[0 0 0 0]
    #  [0 0 0 0]]
    # 然后进行矩阵减法运算得差值矩阵(矩阵对应点相减), inX = [0, 0]时的计算结果为
    # [[-1. - 1.1]
    #  [-1. - 1.]
    #  [0.   0.]
    #  [0. - 0.1]]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 对差值矩阵进行平方运算(矩阵各点计算平方), inX = [0, 0]时的计算结果为
    # [[1.    1.21]
    #  [1.    1.]
    #  [0.    0.]
    #  [0.    0.01]]
    sqDiffMat = diffMat**2
    # 对差值平方矩阵按行进行加法运算得平方和, 即方差, inX = [0, 0]时的计算结果为[ 2.21  2.    0.    0.01]
    sqDistances = sqDiffMat.sum(axis=1) # 方差
    # 对方差进行开方运算得标准差(矩阵各点计算开方), inX = [0, 0]时的计算结果为[ 1.48660687  1.41421356  0.          0.1       ]
    distances = sqDistances**0.5 # 标准差

    # argsort()函数是将distances数组中的元素从小到大排序, 然后提取排序前对应的index(索引)再输出
    # inX = [0, 0]时的输出结果为[2 3 1 0]
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 距离最小的第i个点的index(索引)对应的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 相同的标签求count保存到字典classCount中
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # iteritems()返回字典项的迭代器
    # sorted(iterable, cmp=None, key=None, reverse=False):
    # iterable: 待排序的迭代器;
    # cmp: 比较函数;
    # key: 迭代器中的某个元素作为关键字, operator.itemgetter(index)返回迭代器中索引对应的元素;
    # reverse: 排序规则, reverse=True表示降序, reverse=False表示升序, 默认值为False.
    # 返回排序后的元祖列表
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # 返回发生频率最高的元素标签

def file2matrix(filename):
    '''
    将文本记录转换为NumPy的解析程序
    :param filename: 文件路径
    :return: 特征矩阵和类标签向量
    '''
    fr = open(filename) # 打开文件
    arrayOLines = fr.readlines() # 读取文件所有行
    numberOfLines = len(arrayOLines) # 得到文件行数
    returnMat = np.zeros((numberOfLines, 3)) # 创建以0填充的NumPy矩阵(实际为二维数组), 矩阵行数为文件行数, 列数为3
    classLabelVector = [] # 创建类标签向量
    index = 0 # 行索引, 从0开始
    # 遍历所有行数据
    for line in arrayOLines:
        line = line.strip() # 截取掉字符串的回车字符
        listFromLine = line.split('\t') # 使用tab字符 '\t' 将行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0:3] # 将元素列表的前3个元素存储到矩阵的第index行
        classLabelVector.append(int(listFromLine[-1])) # 将元素列表的最后1个元素转换为int类型并存储到类标签向量
        index += 1 # 行索引index+1
    return returnMat, classLabelVector # 返回特征矩阵和类标签向量

def autoNorm(dataSet):
    '''
    归一化特征值
    将数字特征值转化为0到1的区间, 转换公式为: newValue = (oldValue-min)/(max-min)
    :param dataSet: 待归一化特征值的样本数据
    :return: 数值归一化后的样本数据矩阵、取值范围数组和最小值数组
    '''
    # m, n = np.shape(dataSet)[0], np.shape(dataSet)[1] # (m,n)矩阵
    minVals = dataSet.min(0) # 参数0指示函数从列中选取最小值, 而不是选取当前行的最小值, 结果为(1,n)矩阵
    maxVals = dataSet.max(0) # 参数0指示函数从列中选取最大值, 而不是选取当前行的最大值, 结果为(1,n)矩阵
    ranges = maxVals - minVals # 结果为(1,n)矩阵
    normDataSet = np.zeros(np.shape(dataSet)) # 创建以0填充的同dataSet一样大小的矩阵
    m = dataSet.shape[0] # 样本数据集的行数
    normDataSet = dataSet - np.tile(minVals, (m, 1)) # 计算oldValue-min
    normDataSet = normDataSet / np.tile(ranges, (m, 1)) # 计算(oldValue-min)/(max-min)
    return normDataSet, ranges, minVals # 返回数值归一化后的样本数据矩阵、取值范围数组和最小值数组

def datingClassTest():
    '''
    分类器针对约会网站的测试代码
    :return:
    '''
    hoRatio = 0.10 # 测试数据比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat) # 将数据转换为归一化特征值
    m, n = normMat.shape[0], normMat.shape[1] # (m,n)矩阵
    numTestVecs = int(m*hoRatio) # 测试数据量, normMat[0, numTestVecs]为测试样本数据, normMat[numTestVecs, m]为训练样本数据
    errorCount = 0.0 # 错误数
    # 遍历测试样本文件
    for i in range(numTestVecs):
        # 使用原始kNN分类器预测测试数据的分类
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0 # 预测分类与实际分类不符时错误数errorCount+1
    print "the total error rate is: %f" % (errorCount/float(numTestVecs)) # 输出错误率

def classifyPerson():
    '''
    约会网站预测函数
    :return:
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    # raw_input(): 该函数在标准输出窗口输出参数中的文本内容并返回用户所输入的内容
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per week?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat) # 将数据转换为归一化特征值
    inArr = np.array([ffMiles, percentTats, iceCream])
    # 使用原始kNN分类器预测输入数据的分类
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print 'You will probably like this person: ', resultList[classifierResult - 1]

def img2vector(filename):
    '''
    :param filename: 文件路径
    :return:
    '''
    returnVect = np.zeros((1, 1024)) # 创建以0填充的(1,1024)矩阵
    fr = open(filename) # 打开文件
    # 遍历文件的前32行
    for i in range(32):
        lineStr = fr.readline() # 读取文件一行
        # 遍历当前行的前32列
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j]) # 将(32,32)的矩阵转换为(1,1024)的矩阵
    return returnVect

def handwritingClassTest():
    '''
    手写数字识别系统的测试代码
    :return:
    '''
    hwLabels = [] #手写数字标签向量
    trainingFileList = listdir('trainingDigits') # 获取训练数据集目录的文件列表
    m = len(trainingFileList) # 获取训练数据集的文件数量
    trainingMat = np.zeros((m, 1024)) # 创建以0填充的(m,1024)矩阵
    # 遍历训练样本文件
    for i in range(m):
        fileNameStr = trainingFileList[i] # 取得文件名
        fileStr = fileNameStr.split('.')[0] # 截取掉文件名的(.扩展名)
        classNumStr = int(fileStr.split('_')[0]) # 取得文件表示的真实数字
        hwLabels.append(classNumStr) # 将取得的真实数字添加到标签向量
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr) # 读取文件内容并加入训练样本集
    testFileList = listdir('testDigits') # 获取测试数据集目录的文件列表
    errorCount = 0.0 # 错误数
    mTest = len(testFileList) # 获取测试数据集的文件数量
    # 遍历测试样本文件
    for i in range(mTest):
        fileNameStr = testFileList[i] # 取得文件名
        fileStr = fileNameStr.split('.')[0] # 截取到文件名的(.扩展名)
        classNumStr = int(fileStr.split('_')[0]) # 取得文件表示的真实数字
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) # 读取文件内容作为测试样本数据
        # 使用原始kNN分类器预测测试数据的分类(即识别手写数字)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0 # 预测数字与实际数字不符时错误数errorCount+1
    print "\nthe total number of errors is: %d" % errorCount # 输出错误数
    print "\nthe total error rate is: %f" % (errorCount/float(mTest)) # 输出错误率

if __name__ == '__main__':
    group, labels = createDataSet()
    # print 'group = \n', group
    # print 'labels = \n', labels

    # inX = [0, 0]
    # k = 3
    # label = classify0(inX, group, labels, k)
    # print 'label = ', label

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # print 'datingDataMat = \n', datingDataMat
    # print 'datingLabels = \n', datingLabels

    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # print 'normMat = \n', normMat
    # print 'ranges = \n', ranges
    # print 'minVals = \n', minVals

    # datingClassTest()

    # classifyPerson()

    handwritingClassTest()

    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(facecolor='w') # 创建新的figure, 设置背景色为white
    ax = fig.add_subplot(111) # 添加Axes, 参数: 子图行数, 子图列数, 当前子图位置
    # 绘制散点图, x轴是'玩视频游戏所耗时间百分比', y轴是'每周消费的冰淇淋公升数'
    # 后面2个参数利用变量datingLabels存储的类标签属性, 在散点图上绘制了色彩不等、尺寸不同的点
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    # plt.xlabel(u'玩视频游戏所耗时间百分比', fontsize=15)
    # plt.ylabel(u'每周消费的冰淇淋公升数', fontsize=15)
    # 绘制散点图, x轴是'每年获取的飞行常客里程数', y轴是'玩视频游戏所耗时间百分比'
    # 后面2个参数利用变量datingLabels存储的类标签属性, 在散点图上绘制了色彩不等、尺寸不同的点
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    plt.xlabel(u'每年获取的飞行常客里程数', fontsize=15)
    plt.ylabel(u'玩视频游戏所耗时间百分比', fontsize=15)
    plt.title(u'约会数据散点图', fontsize=18)
    plt.grid(True) # 显示网格
    plt.show() # 显示图表