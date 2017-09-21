#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def loadExData():
    return [[1, 1, 1, 0, 0],
             [2, 2, 2, 0, 0],
             [1, 1, 1, 0, 0],
             [5, 5, 5, 0, 0],
             [1, 1, 0, 2, 2],
             [0, 0, 0, 3, 3],
             [0, 0, 0, 1, 1]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def euclidSim(inA, inB):
    '''
    使用欧式距离计算相似度(1/(1+欧式距离))
    :param inA: 列向量A
    :param inB: 列向量B
    :return: 相似度
    '''
    # np.linalg.norm(): 计算范数的方法, 默认是2范数
    # 向量A减去向量B, 再求向量差的2范数, 就得到了向量A和向量B的欧式距离.
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))

def pearsSim(inA, inB):
    '''
    使用皮尔逊相关系数计算相似度(0.5+0.5*np.corrcoef())
    :param inA: 列向量A
    :param inB: 列向量B
    :return: 相似度
    '''
    if len(inA) < 3: # 小于3个点则完全相关, 相似度为1.0
        return 1.0
    # np.corrcoef(): 计算皮尔逊相关系数, rowvar等于0, 说明传入的数据每一行代表一个样本; 不等于0, 说明传入的数据每一列代表一个样本.
    # 通过0.5+0.5*corrcoef()这个计算公式将取值范围归一化为0到1之间.
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    '''
    计算余弦相似度
    :param inA: 列向量A
    :param inB: 列向量B
    :return: 余弦相似度
    '''
    # 计算两个向量的内积, 即点乘加和
    num = float(inA.T * inB)
    # 计算两个向量的2范数再相乘
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    # 通过0.5+0.5*cos(theta)这个计算公式将取值范围归一化为0到1之间.
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    '''
    在给定相似度计算方法的条件下, 计算用户user对物品item的估计评分值.
    遍历全部物品, 在用户user对某一物品有评分的情况下,计算该物品与物品item的相似度
    (基于对这2个物品都做过评分的所有用户给出的评分值计算相似度)
    得到某个物品与物品item的相似度以后, 再乘以用户user对于该物品的评分, 可以看作是
    用户对于该物品与物品item针对它们相似度的评分, 将这些评分进行累加, 再除以所有物品
    的相似度之和, 就得到了用户对于物品item的估计评分值.
    :param dataMat: 数据矩阵, 行对应用户, 列对应物品, 则行与行之间比较的是基于用户的相似度,
                     列与列之间比较的是基于物品的相似度.
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号
    :return: 用户对于物品item的估计评分值
    '''
    n = np.shape(dataMat)[1] # 物品数目
    simTotal = 0.0 # 总的相似度
    ratSimTotal = 0.0 # (相似度*评分)之和
    # 遍历每一个物品
    for j in range(n):
        userRating = dataMat[user, j] # 得到用户对该物品的评分值
        if userRating == 0: # 如果评分值为0, 就意味着用户没有对该物品评分, 那么就不计算这个物品
            continue
        # dataMat[:, item].A > 0: 编号为item的物品中有评分值(即评分值>0)的数据
        # dataMat[:, j].A > 0: 编号为j(j为遍历的每一个物品)的物品中有评分值(即评分值>0)的数据
        # np.logical_and(True, False) --> False
        # np.logical_and(False, False) --> False
        # np.logical_and(True, True) --> True
        # np.logical_and([True, False, True], [False, False, True]) --> array([False, False, True], dtype=bool)
        # 编号为item的物品和编号为j的物品都有评分值(即评分值>0)的那些行, 即用户
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0: # 如果没有任何一个用户对这2个物品同时评过分, 则这2个物品的相似度为0
            similarity = 0
        else: # 使用指定的相似度计算方法计算这2个物品的相似度(基于用户评分的相似度计算方法)
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print 'the %d and %d similarity is : %f' % (item, j, similarity)
        simTotal += similarity # 物品item和物品j的相似度做累加
        ratSimTotal += similarity * userRating # 物品item和物品j的相似度乘以用户对物品j的评分, 再做累加
    if simTotal == 0: # 相似度之和为0, 则用户对于物品item的估计评分值也为0
        return 0
    else: # 相似度之和非0
        return ratSimTotal / simTotal # 返回用户对于物品item的估计评分值

def svdEst(dataMat, user, simMeas, item):
    '''
    在给定相似度计算方法的条件下, 计算用户user对物品item的估计评分值.
    本函数是基于SVD进行估计评分的.
    :param dataMat: 数据矩阵, 行对应用户, 列对应物品, 则行与行之间比较的是基于用户的相似度,
                     列与列之间比较的是基于物品的相似度.
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号
    :return: 用户对于物品item的估计评分值
    '''
    n = np.shape(dataMat)[1] # 物品数目
    simTotal = 0.0 # 总的相似度
    ratSimTotal = 0.0 # (相似度*评分)之和
    # 执行SVD, Sigma是NumPy数组的形式
    U, Sigma, VT = np.linalg.svd(dataMat)
    # 使用包含90%能量值的奇异值建立对角矩阵
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    # dataMat: mxn, 行对应用户, 列对应物品
    # dataMat.T: nxm, dataMat转置之后, 行对应物品, 列对应用户
    # U: mxm, 则U[:, :4]: mx4
    # Sig4: 4x4
    # U矩阵会将物品映射到低维空间中, VT矩阵会将用户映射到低维空间中
    # 计算得到的矩阵, nx4, 行仍对应物品, 列仍对应用户, 物品总数未变, 减少的是用户
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    # 遍历每一个物品
    for j in range(n):
        userRating = dataMat[user, j] # 得到用户对该物品的评分值
        if userRating == 0 or j == item: # 如果评分值为0, 就意味着用户没有对该物品评分, 或者物品j和物品item是同一件物品, 那么就不计算这个物品
            continue
        # 在低维空间下, 使用指定的相似度计算方法计算这2个物品的相似度(基于用户评分的相似度计算方法)
        # 矩阵xformedItems的行对应物品, 列对应用户, 相似度计算方法的参数为列向量, 所以
        # xformedItems[item, :].T 是物品item的列向量,
        # xformedItems[j, :].T 是物品j的列向量.
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print 'the %d and %d similarity is : %f' % (item, j, similarity)
        simTotal += similarity # 物品item和物品j的相似度做累加
        ratSimTotal += similarity * userRating # 物品item和物品j的相似度乘以用户对物品j的评分, 再做累加
    if simTotal == 0: # 相似度之和为0, 则用户对于物品item的估计评分值也为0
        return 0
    else: # 相似度之和非0
        return ratSimTotal / simTotal # 返回用户对于物品item的估计评分值

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    '''
    基于物品相似度的推荐引擎
    :param dataMat: 数据矩阵, 行对应用户, 列对应物品, 则行与行之间比较的是基于用户的相似度,
                     列与列之间比较的是基于物品的相似度
    :param user: 用户编号
    :param N: 推荐物品的数量
    :param simMeas: 相似度计算方法
    :param estMethod: 评分估计方法
    :return:
    '''
    # 寻找用户未评分的物品, 建立未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: # 不存在用户未评分的物品则直接退出
        return 'you rated everything'
    itemScores = [] # 估计评分值列表
    for item in unratedItems: # 遍历所有的未评分物品
        # 计算用户对于每个未评分物品的估计评分值
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 将物品编号和用户对该物品的估计评分值保存到估计评分值列表中
        itemScores.append((item, estimatedScore))
    # 对估计评分值列表按照估计评分值进行从大到小的排序, 返回指定的N个物品
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def printMat(inMat, thresh=0.8):
    '''
    打印矩阵
    :param inMat: 矩阵
    :param thresh: 阈值
    :return:
    '''
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1, # 矩阵元素大于阈值时打印1
            else:
                print 0, # 矩阵元素不大于阈值时打印0
        print ''

def imgCompress(numSV=3, thresh=0.8):
    '''
    压缩图像, 再基于任意给定的奇异值数目来重构图像
    :param numSV: 重构图像所需要的奇异值数目
    :param thresh: 阈值
    :return:
    '''
    my1 = [] # 原数据集
    # 读文件字符存入原数据集
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        my1.append(newRow)
    myMat = np.mat(my1)
    print '*****original matrix*****'
    printMat(myMat, thresh)
    # 对原始图像进行SVD分解
    U, Sigma, VT = np.linalg.svd(myMat) # Sigma是一个NumPy数组, 它实际存储的是对角矩阵对角线上的值
    # 建立全0矩阵, 矩阵的行数和列数都是参数numSV给定的奇异值数目
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    # 将Sigma里的奇异值填充到SigRecon的对角线上
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    # 通过截断U和VT矩阵, 用SigRecon得到重构后的矩阵
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print '*****reconstructed matrix using %d singular values*****' % numSV
    printMat(reconMat, thresh)

if __name__=='__main__':
    np.set_printoptions(linewidth=300)
    # Data = loadExData()
    # # 执行SVD
    # U, Sigma, VT = np.linalg.svd(Data)
    # print 'U: \n', U
    # print 'Sigma: \n', Sigma
    # print 'VT: \n', VT
    #
    # percents = Sigma**2 / sum(Sigma**2)
    # print percents
    # mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    # mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    # fig = plt.figure()
    # ind = range(len(percents))
    # plt.scatter(ind, percents, marker='o', alpha=1, s=50)
    # plt.plot(ind, percents)
    # plt.ylabel(u'Sigma的平方百分比')
    # plt.show()
    #
    # # Sigma的前3个数值比其它2个数值大了很多, 使用前3个数值构建一个3x3的矩阵
    # Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # # 重构原始矩阵的近似矩阵
    # Data1 = U[:, :3] * Sig3 * VT[:3, :]
    # print '原始矩阵: \n', np.mat(Data)
    # print '近似矩阵: \n', Data1

    # myMat = np.mat(loadExData())
    # euclidS = euclidSim(myMat[:, 0], myMat[:, 4])
    # print euclidS
    # euclidS = euclidSim(myMat[:, 0], myMat[:, 0])
    # print euclidS
    # cosS = cosSim(myMat[:, 0], myMat[:, 4])
    # print cosS
    # cosS = cosSim(myMat[:, 0], myMat[:, 0])
    # print cosS
    # pearsS = pearsSim(myMat[:, 0], myMat[:, 4])
    # print pearsS
    # pearsS = pearsSim(myMat[:, 0], myMat[:, 0])
    # print pearsS

    # myMat = np.mat([[4,4,0,2,2],[4,0,0,3,3],[4,0,0,1,1],[1,1,1,2,0],[2,2,2,0,0],[1,1,1,0,0],[5,5,5,0,0]])
    # print myMat
    # rec = recommend(myMat, 2)
    # print rec
    # rec = recommend(myMat, 2, simMeas=euclidSim)
    # print rec
    # rec = recommend(myMat, 2, simMeas=pearsSim)
    # print rec

    # myMat = np.mat(loadExData2())
    # 执行SVD
    # U, Sigma, VT = np.linalg.svd(myMat)
    # print 'U: \n', U
    # print 'Sigma: \n', Sigma
    # print 'VT: \n', VT
    # Sig2 = Sigma**2

    # rec = recommend(myMat, 1, estMethod=svdEst)
    # print rec
    # rec = recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim)
    # print rec

    imgCompress(2)
