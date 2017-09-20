#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    '''
    读文件加载数据并转化为矩阵
    :param fileName: 文件路径
    :param delim: 分隔符, 默认'tab'符
    :return: 数据矩阵
    '''
    # 只读打开文件
    fr = open(fileName)
    # 读文件使用delim(默认'tab'符)分隔每一行文本生成列表(m行xn列)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # 使用map函数将列表中的每一个数值(字符串类型)转换为float类型
    datArr = [map(float, line) for line in stringArr]
    # 将上述列表转换成为矩阵形式并返回
    return np.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    '''
    pca(Principal Component Analysis, 主成分分析)降维过程
    :param dataMat: 数据矩阵
    :param topNfeat: 需要保留的特征数目
    :return: 转换后的数据矩阵和被重构的原始数据矩阵
    '''
    # 按列求矩阵的均值, 得到1行xn列的行向量
    meanVals = np.mean(dataMat, axis=0)
    # 矩阵每一列的特征值减去该列的均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵, rowvar等于0, 说明传入的数据每一行代表一个样本; 不等于0, 说明传入的数据每一列代表一个样本.
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 对特征值进行从小到大排序并返回对应的索引值
    eigValInd = np.argsort(eigVals)
    # 因从小到大排序, 所以采用逆序方式获取topNfeat个最大特征的索引值
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 获得topNfeat个最大的特征向量, 这些特征向量构成对数据进行转换的矩阵, 即转换矩阵
    redEigVects = eigVects[:, eigValInd]
    # 去除均值后的数据矩阵 * 转换矩阵, 将原始数据转换到新的空间中实现降维, 维度是topNfeat.
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵重构原数据矩阵(用作调试, 可与原始数据矩阵进行比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 返回转换后的数据矩阵和被重构的原始数据矩阵
    return lowDDataMat, reconMat

def replaceNanWithMean():
    '''
    将NaN替换成平均值
    :return: 替换后的数据矩阵
    '''
    # 读文件加载数据, 使用空格分隔每一行, 生成float类型的数据矩阵
    datMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(datMat)[1] # 特征数量
    # 遍历每一个特征, 将NaN替换成该特征下非NaN特征值的均值
    for i in range(numFeat):
        # matrix.A: Return matrix self as an ndarray object.Equivalent to np.asarray(matrix).
        # np.isnan(datMat[:, i].A): 矩阵第i列的某个值是否为NaN, 返回一个列向量, 列向量的每个值为True或者False

        # 假设a是一个数组, 则np.nonzero(a)返回数组a中值不为零的元素的下标, 它的返回值是一个长度为a.ndim(数组a的轴数)的元组,
        # 元组的每个元素都是一个整数数组, 其值为非零元素的下标在对应轴上的值.
        # 对于一维布尔数组a1, np.nonzero(a1)得到一个长度为1的元组, 如下例所示它表示a1[0]和a1[2]的值不为0(False)
        # >>> a1 = np.array([True, False, True, False])
        # >>> np.nonzero(a1)
        # (array([0, 2]),)
        # 对于二维数组a2, np.nonzero(a2)得到一个长度为2的元组. 它的第0个元素是数组a2中值不为0的元素的第0轴的下标,
        # 第1个元素是数组a2中值不为0的元素的第1轴的下标, 如下例所示它表示a2[0, 0], a2[0, 2]和a2[1, 0]的值不为0(False)
        # >>> a2 = np.array([[True, False, True], [True, False, False]])
        # >>> np.nonzero(a2)
        # (array([0, 0, 1]), array([0, 2, 0]))

        # 计算矩阵第i列所有非NaN的平均值
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        # 将矩阵第i列的所有NaN置为上面计算得到的平均值
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat

if __name__=='__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print np.shape(lowDMat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90, c='b')
    # ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.plot(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], c='black')
    plt.show()

    np.set_printoptions(linewidth=300, edgeitems=20)
    # 确认所需特征和可以去除特征的数目
    # 将数据集中所有的NaN替换成平均值
    dataMat = replaceNanWithMean()
    # 按列求矩阵的均值, 得到1行xn列的行向量
    meanVals = np.mean(dataMat, axis=0)
    # 矩阵每一列的特征值减去该列的均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵, rowvar等于0, 说明传入的数据每一行代表一个样本; 不等于0, 说明传入的数据每一列代表一个样本.
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print eigVals
    percents = eigVals/sum(eigVals)
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure()
    ind = range(0, 20)
    plt.plot(ind, percents[:len(ind)])
    plt.ylabel(u'方差的百分比')
    plt.show()
