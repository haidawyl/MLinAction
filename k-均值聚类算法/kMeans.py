#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import urllib
import json
from time import sleep

def loadDataSet(fileName):
    '''
    加载文件数据
    :param fileName: 文件路径
    :return:
    '''
    dataMat = [] # 数据矩阵
    fr = open(fileName) # 打开文件
    for line in fr.readlines(): # 遍历文件的每一行
        curLine = line.strip().split('\t') # 截取掉每行的回车字符, 再使用tab字符 '\t' 将行数据分割成一个元素列表
        # map()会根据提供的函数对指定序列做映射. 第一个参数function以参数序列中的每一个元素调用function函数, 返回包含每次function函数返回值的新列表.
        fltLine = map(float, curLine) # 将curLine列表中元素转换成float类型
        dataMat.append(fltLine) # 加入数据矩阵
    return dataMat

def distEclud(vecA, vecB):
    '''
    计算向量A和向量B的欧式距离即标准差(方差的平方根)
    :param vecA: 向量A
    :param vecB: 向量B
    :return: 返回向量A和向量B的标准差
    '''
    # np.power(x1, x2): 求x1的x2次方
    # np.power(2, 3) = 8
    # np.power([0, 1, 2, 3, 4, 5], 3) = array([  0,   1,   8,  27,  64, 125])
    # np.power([0, 1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]) = array([  0.,   1.,   8.,  27.,  16.,   5.])
    # np.sum([0.5, 1.5]) = 2.0
    # np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32) = 1
    # np.sum([[0, 1], [0, 5]]) = 6
    # np.sum([[0, 1], [0, 5]], axis=0) = array([0, 6]) 按列加和
    # np.sum([[0, 1], [0, 5]], axis=1) = array([1, 5]) 按行加和
    # (1)计算向量A和向量B的差, 即向量各元素的差
    # (2)计算各元素差的平方
    # (3)计算各元素差的平方的和
    # (4)计算和值的平方根
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    '''
    为给定数据集构建一个包含k个随机质心的集合. 随机质心必须要在整个数据集的边界之内,
    可以通过找到数据集每一维的最小值和最大值来完成. 通过生成0到1.0之间的随机数并乘以
    取值范围再加上最小值来获得, 以便确保随机点在数据的边界之内.
    :param dataSet: 数据集
    :param k: 簇的数目
    :return:
    '''
    warnings.simplefilter("ignore")

    n = np.shape(dataSet)[1] # 数据集的列数
    centroids = np.mat(np.zeros((k, n))) # 创建以0填充的k行xn列的簇质心矩阵
    for j in range(n): # 遍历数据集的所有列
        minJ = min(dataSet[:, j]) # 取得第j列的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ) # 取得第j列最大值与最小值的差
        # np.random.rand(): 创建一个给定类型的数组，将其填充在一个均匀分布的随机样本[0, 1)中
        # np.random.rand(k, 1): 创建一个k行x1列的数组，将其填充在一个均匀分布的随机样本[0, 1)中
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1) # 构建簇质心
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    k-均值聚类算法
    :param dataSet: 数据集
    :param k: 期望的簇数目
    :param distMeas: 计算距离的函数
    :param createCent: 创建初始质心的函数
    :return:
    '''
    m = np.shape(dataSet)[0] # 矩阵行数
    clusterAssment = np.mat(np.zeros((m, 2))) # 创建以0填充的m行x2列的矩阵即簇分配结果矩阵[簇质心索引值, 误差(当前点到簇质心的距离)]
    centroids = createCent(dataSet, k) # 根据给定数据集构建一个包含k个随机簇质心的集合.
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 更改每一行数据所属的簇, 规则为找到离数据点最近的簇质心, 则该数据点属于这个簇
        for i in range(m): # 遍历数据集的每一行
            minDist = np.inf # 最小误差即当前数据到各个簇质心距离的最小值, 默认为无穷大
            minIndex = -1 # 误差最小的簇质心的索引值
            # 寻找离数据集第i行距离最小的簇质心
            for j in range(k): # 遍历簇质心集合的每一行
                distJI = distMeas(centroids[j, :], dataSet[i, :]) # 计算数据集第i行和簇质心集合第j行的距离
                if distJI < minDist: # 寻找距离最小的簇质心
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: # 当前数据点所属的簇质心非离它最近的簇质心时将会更改该数据点所属的簇, 即任一点的簇分配结果发生改变则外层循环继续
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2 # 更新簇分配结果矩阵即更改当前数据点的簇质心索引值和误差(当前数据集与所属簇的质心之间距离的平方)
        # print centroids
        # 重新计算所有簇质心的取值
        for cent in range(k): # 遍历所有簇质心
            # 通过数组过滤来获得给定簇的所有点
            # clusterAssment[:, 0].A == cent: 簇分配结果矩阵中的簇质心索引值与当前遍历的簇质心索引值相等则返回True即非0, 不相等则返回False即0
            # np.nonzero(clusterAssment[:, 0].A == cent): 取得非0的簇分配结果矩阵的数据集索引值即属于当前簇的数据集索引值
            # dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]: 根据索引值取得所有数据即取得数据集中属于当前簇的所有数据
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # X = np.array([[1, 2], [4, 5], [7, 8]])
            # np.mean(X, axis=0) = [ 4.  5.] 取每一列的平均值
            # np.mean(X, axis=1) = [ 1.5  4.5  7.5] 取每一行的平均值
            # 计算所有点的均值, 选项axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = np.mean(ptsInClust, axis=0) # 重新计算当前簇质心的取值
    return centroids, clusterAssment # 返回簇质心列表和簇分配结果

def biKmeans(dataSet, k, distMeas=distEclud):
    '''
    二分k-均值聚类算法
    :param dataSet: 数据集
    :param k: 期望的簇数目
    :param distMeas: 计算距离的函数
    :return:
    '''
    m = np.shape(dataSet)[0] # 矩阵行数
    clusterAssment = np.mat(np.zeros((m, 2))) # 创建以0填充的m行x2列的矩阵即簇分配结果矩阵[簇质心索引值, 误差(当前点到簇质心的距离)]
    # 计算全部数据集的均值, 选项axis=0表示沿矩阵的列方向进行均值计算
    # 初始时将整个数据集看成一个簇, 计算该簇的质心
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0] # 存储所有簇质心的列表, 初始值仅包含整个数据集的簇质心
    # 第一次分配簇, 即计算整个数据集到初始簇质心的误差值
    for j in range(m): # 遍历每一行数据
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2 # 计算每一行数据到初始簇质心的误差值
    while (len(centList) < k): # 循环划分簇, 直到得到想要的簇数目为止.
        lowestSSE = np.inf # 最小的SSE(Sum of Squared Error, 即误差平方和), 默认为无穷大
        # 对每一个簇进行一次k-均值(k=2)聚类, 即将其一分为二, 找到使得划分之后的SSE
        # (等于簇划分之后的SSE+未划分的簇的SSE)最小的簇进行实际的划分操作
        for i in range(len(centList)): # 遍历簇列表中的每一个簇
            # clusterAssment[:, 0].A == i: 簇分配结果矩阵中的簇质心索引值与当前遍历的簇质心索引值相等则返回True即非0, 不相等则返回False即0
            # np.nonzero(clusterAssment[:, 0].A == i): 取得非0的簇分配结果矩阵的数据集索引值即属于当前簇的数据集索引值、
            # dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]: 根据索引值取得所有数据即取得数据集中属于当前簇的所有数据的所有列形成的矩阵
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 将属于当前簇的数据集划分到2个簇, 同时给出每个簇的误差值
            sseSplit = np.sum(splitClustAss[:, 1]) # 对划分后生成的新簇中的误差求和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1]) # 对未划分簇(即不属于当前簇)的误差求和
            print "sseSplit and sseNotSplit", sseSplit, sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE: # 这两个误差之和作为本次划分的误差, 如果本次划分的误差小于当前最小的误差, 则本次划分被保存
                bestCentToSplit = i # 当前的簇索引为最优的划分簇索引
                bestNewCents = centroidMat # 当前的簇质心集合为最优的簇质心集合
                bestClustAss = splitClustAss.copy() # 当前的簇分配结果为最优的簇分配结果
                lowestSSE = sseSplit + sseNotSplit # 本次划分后的误差为当前最小的误差
        # 使用kMeans()函数并指定簇数目为2来划分当前簇, 会得到两个编号分别为0和1的结果簇.
        # 需要将这些簇编号修改为划分前簇的编号和新加簇的编号
        # bestClustAss[:, 0].A == 1: 最优的簇分配结果中的簇质心索引值等于1返回True即非0, 不等于1返回False即0
        # np.nonzero(bestClustAss[:, 0].A == 1): 取得最优的簇分配结果中簇质心索引值为1的全部数据在数据集中的索引值
        # bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0]: 取得最优的簇分配结果中簇质心索引值为1的簇质心索引值
        # 将最优的簇分配结果中值为1的簇质心索引值修改为簇质心列表的长度即当前簇质心列表中最大的索引值+1作为新划分出来的数据集的簇质心索引值,
        # 也就是划分后新的数据集的簇质心索引值更新为原簇质心列表中没有的新的索引值
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 将最优的簇分配结果中值为0的簇质心索引值修改为被划分的簇质心索引值, 即划分后属于原来簇的数据集恢复原簇质心索引值
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0] # 划分前原簇质心更新为划分后簇质心索引值为0的簇质心
        centList.append(bestNewCents[1, :].tolist()[0]) # 划分后簇质心索引值为1的簇质心即划分出来的新簇质心加入到簇质心列表中
        # 划分前原簇的分配结果更新为划分后新簇的分配结果
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment # 返回簇质心列表和簇分配结果

def geoGrab(stAddress, city):
    '''
    查询Yahoo! API获得地理位置信息
    :param stAddress: 地址
    :param city: 城市
    :return:
    '''
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.load(c.read())

def massPlaceFind(fileName):
    '''
    根据文件中的地址访问Yahoo! API获得经纬度信息, 然后将地址和经纬度保存到新的文件中
    :param fileName:
    :return:
    '''
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude']) # 纬度
            lng = float(retDict['ResultSet']['Results'][0]['longitude']) # 经度
            print '%s\t%f\t%f' % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng)) # 地址\t纬度\t经度
        else:
            print 'error fetching'
        sleep(1)
    fw.close()

def testKmeans(k):
    '''
    :return:
    '''
    dataMat = np.mat(loadDataSet('testSet.txt'))

    myCentroids, clustAssing = kMeans(dataMat, k)
    print 'myCentroids:\n', myCentroids
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    plt.scatter(np.array(myCentroids[:, 0]), np.array(myCentroids[:, 1]), marker='+', alpha=1, s=150)
    for cent in range(k):
        ptsInClust = dataMat[np.nonzero(clustAssing[:, 0].A == cent)[0]]
        plt.scatter(np.array(ptsInClust[:, 0]), np.array(ptsInClust[:, 1]), marker='o', alpha=1)
    plt.show()

def testBiKmeans(k):
    '''
    :param k:
    :return:
    '''
    dataMat = np.mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(dataMat, k)
    print 'centList:\n', centList
    plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    plt.scatter(np.array(centList[:, 0]), np.array(centList[:, 1]), marker='+', alpha=1, s=150)
    for cent in range(k):
        ptsInClust = dataMat[np.nonzero(myNewAssments[:, 0].A == cent)[0]]
        plt.scatter(np.array(ptsInClust[:, 0]), np.array(ptsInClust[:, 1]), marker='o', alpha=1)
    plt.show()

def distSLC(vecA, vecB):
    '''
    计算地球表面两点之间的距离, 单位是英里.
    给定两个点的经纬度, 可以使用球面余弦定理来计算两点的距离.
    这里的纬度和经度用角度作为单位, 但是np.sin()和np.cos()以弧度作为输入.
    可以将角度除以180然后再乘以圆周率np.pi转换为弧度.
    :param vecA:
    :param vecB:
    :return:
    '''
    # vecA[0, 1]: 纬度
    # vecA[0, 0]: 经度
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0 # 反余弦

def clusterClubs(numClust=5):
    '''
    将文本文件中的俱乐部进行聚类并画出结果
    :param numClust: 期望的簇数目
    :return:
    '''
    datList = [] # 经纬度数据列表
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])]) # 添加[经度, 纬度]
    datMat = np.mat(datList) # 转换为矩阵
    # 使用二分k-均值聚类算法, 计算距离的方法为distSLC(), 即计算地球表面两点之间的距离
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8] # 该矩形决定绘制图的哪一部分
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<'] # 标记形状列表用于绘制散点图
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops) # 绘制地图
    imgP = plt.imread('Portland.png') # 基于图像创建矩阵
    ax0.imshow(imgP) # 绘制图像矩阵
    ax1 = fig.add_axes(rect, label='ax1', frameon=False) # 绘制另一幅图, 允许使用两套坐标系统且不用做任何缩放或者偏移
    for i in range(numClust): # 遍历每一个簇并绘制簇下的数据点
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :] # 取得经纬度数据
        markerStyle = scatterMarkers[i % len(scatterMarkers)] # 选择标记形状
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90) # 绘制经纬度数据
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300) # 绘制簇质心
    plt.show()

if __name__=='__main__':
    # testKmeans(4)
    # testBiKmeans(3)
    clusterClubs()


