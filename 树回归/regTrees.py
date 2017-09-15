#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(fileName):
    '''
    :param fileName:
    :return:
    '''
    dataMat = [] # 数据矩阵
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine) # 将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    '''
    切分数据集
    :param dataSet: 待切分的数据集
    :param feature: 切分数据集的特征索引
    :param value: 切分数据集的特征值
    :return: 返回切分后的两个子集
    '''
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0]] # 将数据集最佳切分特征上值大于最佳切分特征值的数据分到左子集
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0]] # 将数据集最佳切分特征上值不大于最佳切分特征值的数据分到右子集
    return mat0, mat1 # 返回左右两个子集

def regLeaf(dataSet):
    '''
    生成回归树叶节点, 该叶节点只存储目标变量的均值
    :param dataSet: 数据集
    :return: 数据集目标变量的均值
    '''
    return np.mean(dataSet[:, -1]) # 返回数据集目标变量的均值

def regErr(dataSet):
    '''
    计算数据集目标变量的总方差
    :param dataSet: 数据集
    :return: 数据集目标变量的总方差
    '''
    # 数据集目标变量的均方差再乘以数据个数即得到数据集目标变量的总方差
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0] # 返回数据集目标变量的总方差

def createTree(dataSet, leafType=regLeaf, errTpye=regErr, ops=(1,4)):
    '''
    生成树
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数
    :param errTpye: 误差计算函数
    :param ops: 生成树所需其它参数的元祖
    :return: 树的字典
    '''
    # 找到最佳的二元切分方式. 如果feat为None, 则val是叶节点的模型
    # (在回归树中,该模型就是目标变量的均值;而在模型树中,该模式是一个线性方程)
    # 否则feat是最佳切分方式的特征索引, val是最佳切分方式的特征值.
    feat, val = chooseBestSplit(dataSet, leafType, errTpye, ops)
    if feat == None: # 切分特征索引为空时, 则无需再切分而是直接返回叶节点值
        return val
    retTree = {} # 保存树的字典
    retTree['spInd'] = feat # 最佳切分方式的特征索引
    retTree['spVal'] = val # 最佳切分方式的特征值
    # 根据最佳切分方式切分当前的数据集, 生成左右两棵子树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errTpye, ops) # 递归创建左子树
    retTree['right'] = createTree(rSet, leafType, errTpye, ops) # 递归创建右子树
    return retTree # 返回树的字典

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    找到数据集的最佳二元切分方式, 如果找不到, 则返回None和当前数据生成的叶节点;
    如果能够找到, 则返回特征索引和切分特征值. "最佳"意味着总方差最小.
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 生成树所需其它参数的元祖
    :return:
    '''
    tolS = ops[0] # 用户设置的容许误差下降值
    tolN = ops[1] # 用户设置的最少样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1: # 如果所有值相等就不需要再切分而直接返回
        return None, leafType(dataSet)
    m, n = np.shape(dataSet) # 当前数据集大小
    S = errType(dataSet) # 当前数据集的总方差
    bestS = np.inf # 最小的总方差, 默认为正无穷大
    bestIndex = 0 # 最佳切分特征的索引
    bestValue = 0 # 最佳切分的特征值
    # 遍历每一个特征和该特征下的所有特征值, 找到一个最佳的切分方式, 即使得切分后总方差最小切分方式
    for featIndex in range(n-1): # 遍历每一个特征
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]): # 遍历某个特征下的所有特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) # 切分数据集
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): # 如果切分后任意一个数据集的样本数小于用户指定的最少样本数, 则不采取这种切分方式
                continue
            newS = errType(mat0) + errType(mat1) # 计算切分后两个数据子集的总方差之和
            # 保存切分后总方差之和最小的特征索引和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS: # 如果总方差减少量小于用户指定的容许误差下降值, 则不切分数据集
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) # 执行最佳切分方式切分当前的数据集
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): # 如果切分后任意一个子集的样本数小于用户指定的最少样本数, 则不切分数据集
        return None, leafType(dataSet)
    return bestIndex, bestValue

def isTree(obj):
    '''
    判断是否是一颗树, 或者说是否为非叶节点
    :param obj:
    :return:
    '''
    return (type(obj).__name__=='dict')

def getMean(tree):
    '''
    从上往下遍历树直到叶节点, 如果找到两个叶节点就计算它们的平均值.
    当计算完最底层的两个叶节点平均值之后返回给上层节点, 这时该上层节点也变成了叶节点,
    再计算这层叶节点的平均值然后继续返回给上一层, 依此类推, 直至树的根节点.
    这个函数对树进行塌陷处理, 即返回整棵树的平均值.
    :param tree:
    :return: 整棵树的平均值
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    '''
    剪枝
    :param tree: 待剪枝的树
    :param testData: 剪枝所需的测试数据
    :return:
    '''
    if np.shape(testData)[0] == 0: # 没有测试数据则对树进行塌陷处理, 计算整棵树的平均值作为叶节点模型
        return getMean(tree)
    # 非空, 则反复递归调用函数prune()对树进行切分.
    if (isTree(tree['right']) or isTree(tree['left'])): # 如果树的右孩子或者左孩子还是树, 则根据该节点保存的最佳切分方式切分测试数据集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) # 将测试数据集切分成两个数据子集
    if isTree(tree['left']): # 如果树的左孩子也是树, 则根据测试数据集的左子集剪枝左子树
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): # 如果树的右孩子也是树, 则根据测试数据集的右子集剪枝右子树
        tree['right'] = prune(tree['right'], rSet)
    # 在对左右两个子树完成剪枝之后, 还需要检查它们是否仍然是子树, 如果两个分支不再是子树了,
    # 那么判断它们是否可以合并. 如果是的话就合并.具体的做法就是对合并前后的误差进行比较, 如果
    # 合并后的误差比不合并的误差小就进行合并操作, 反之则不合并而直接返回.
    if not isTree(tree['left']) and not isTree(tree['right']): # 如果树的左,右孩子都是叶节点, 则判断是否需要剪枝
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) # 根据该节点保存的最佳切分方式切分测试数据集
        # 测试数据集的左子集减去左叶节点的均值,再平方
        # 测试数据集的右子集减去右叶节点的均值,再平方
        # 左右子集均方误差之和即为切分后(合并前)测试数据集左右子集的总方差之和
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0 # 左右叶节点的均值
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2)) # 计算数据集切分前(合并后)的总方差
        if errorMerge < errorNoMerge: # 如果切分前(合并后)的总方差小于切分后(合并前)的总方差, 说明切分数据集导致总方差增加了, 则需要剪枝, 即合并当前的叶节点.
            print 'merging'
            return treeMean # 返回左右叶节点的均值. 这就是剪枝操作, 减掉叶节点, 叶节点的父节点变成了叶节点.
        else:
            return tree # 不需要剪枝, 返回树本身
    else:
        return tree # 不需要剪枝, 返回树本身

def getNumLeafs(tree):
    '''
    获取叶节点的数目
    :param tree:
    :return:
    '''
    if not isTree(tree['left']) and not isTree(tree['right']): # 左右孩子都是叶节点
        return 2 # 二叉树
    elif isTree(tree['left']) and isTree(tree['right']): # 左右孩子都不是叶节点
        return getNumLeafs(tree['left']) + getNumLeafs(tree['right'])
    elif isTree(tree['left']): # 右孩子是叶节点
        return 1 + getNumLeafs(tree['left'])
    elif isTree(tree['right']): # 左孩子是叶节点
        return 1 + getNumLeafs(tree['right'])

def getLeaves(tree, leaves):
    '''
    获取叶节点
    :param tree:
    :return:
    '''
    if isTree(tree['left']): # 左孩子是树, 则递归调用函数getLeaves()获取叶节点
        getLeaves(tree['left'], leaves)
    else: # 左孩子是叶节点, 则直接加入列表
        leaves.append(tree['left'])
    if isTree(tree['right']): # 右孩子是树, 则递归调用函数getLeaves()获取叶节点
        getLeaves(tree['right'], leaves)
    else: # 右孩子是叶节点, 则直接加入列表
        leaves.append(tree['right'])

def getTreeDepth(tree):
    '''
    获取树的深度, 根节点的深度为1
    :param tree:
    :return:
    '''
    leftMaxDepth = 0 # 左子树最大深度
    rightMaxDepth = 0 # 右子树最大深度
    if isTree(tree['left']): # 如果树的左孩子也是树
        leftMaxDepth = 1 + getTreeDepth(tree['left'])  # 则递归计算左子树的深度再+1作为左子树最大深度
    else:
        leftMaxDepth = 1
    if isTree(tree['right']): # 如果树的右孩子也是树
        rightMaxDepth = 1 + getTreeDepth(tree['right'])  # 则递归计算右子树的深度再+1作为右子树最大深度
    else:
        rightMaxDepth = 1
    return max(leftMaxDepth, rightMaxDepth) # 返回树的最大深度

def linearSolve(dataSet):
    '''
    计算线性回归的回归系数
    :param dataSet: 数据集
    :return: 线性回归的回归系数, 矩阵X, 矩阵Y
    '''
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1] # X的第1列全为1, 其它列是数据集的前n-1列
    Y = dataSet[:, -1] # Y是数据集的最后1列
    xTx = X.T * X
    # 计算行列式, 如果为0则矩阵的逆不存在
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y # 返回线性回归的回归系数, 矩阵X, 矩阵Y

def modelLeaf(dataSet):
    '''
    生成模型树叶节点, 叶节点存储的是线性回归的回归系数
    :param dataSet: 数据集
    :return: 线性回归的回归系数
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws # 返回线性回归的回归系数

def modelErr(dataSet):
    '''
    计算模型树的平方误差
    :param dataSet: 数据集
    :return: 模型树的平方误差
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws # 计算预测值
    return sum(np.power(Y - yHat, 2)) # 返回真实值与预测值的平方误差

def regTreeEval(model, inDat):
    '''
    计算回归树预测值
    :param model: 叶节点模型
    :param inDat: 待预测数据
    :return: 预测值
    '''
    return float(model) # 返回预测值

def modelTreeEval(model, inDat):
    '''
    计算模型树预测值
    :param model: 叶节点模型
    :param inDat: 待预测数据
    :return: 线性回归预测值
    '''
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1))) # 生成矩阵X, 默认全部为1
    X[:, 1:n+1] = inDat # 矩阵X的第2列到第n+1列赋值为待预测数据, 第1列是默认值1.0
    return float(X * model) # 返回线性回归预测值

def treeForeCast(tree, inData, modelEval=regTreeEval):
    '''
    计算数据预测值
    :param tree: 树模型
    :param inData: 待预测数据
    :param modelEval: 计算叶节点预测值的函数, 默认为回归树模型函数
    :return:
    '''
    if not isTree(tree): # 如果是叶节点, 直接返回预测值
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']: # 待预测数据在最佳切分特征上的值大于最佳切分特征值
        if (isTree(tree['left'])): # 树的左孩子还是树, 则递归调用函数treeForeCast预测该数据
            return treeForeCast(tree['left'], inData, modelEval)
        else: # 树的左孩子是叶节点, 则直接返回预测值
            return modelEval(tree['left'], inData)
    else: # 待预测数据在最佳切分特征上的值不大于最佳切分特征值
        if (isTree(tree['right'])): # 树的右孩子还是树, 则递归调用函数treeForeCast预测该数据
            return treeForeCast(tree['right'], inData, modelEval)
        else: # 树的右孩子是叶节点, 则直接返回预测值
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    '''
    计算测试集的预测值
    :param tree: 树模型
    :param testData: 待预测数据集
    :param modelEval: 计算叶节点预测值的函数, 默认为回归树模型函数
    :return:
    '''
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m): # 遍历测试集的每一行数据计算预测值
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

def regTreeTest():
    '''
    测试回归树
    :return:
    '''
    myDat1 = loadDataSet('ex00.txt')
    myMat1 = np.mat(myDat1)
    retTree1 = createTree(myMat1)
    print retTree1
    print '树的深度; ', getTreeDepth(retTree1)
    print '树的叶节点数; ', getNumLeafs(retTree1)
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(myMat1[:, 0].flatten().A[0], myMat1[:, 1].flatten().A[0], s=2, c='r')
    plt.show()

    myDat2 = loadDataSet('ex0.txt')
    myMat2 = np.mat(myDat2)
    retTree2 = createTree(myMat2)
    print retTree2
    print '树的深度; ', getTreeDepth(retTree2)
    print '树的叶节点数; ', getNumLeafs(retTree2)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5))  # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(myMat2[:, 1].flatten().A[0], myMat2[:, 2].flatten().A[0], s=2, c='r')
    plt.show()

    myDat3 = loadDataSet('ex2.txt')
    myMat3 = np.mat(myDat3)
    retTree3 = createTree(myMat3)
    print retTree3
    print '树的深度; ', getTreeDepth(retTree3)
    print '树的叶节点数; ', getNumLeafs(retTree3)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5))  # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(myMat3[:, 0].flatten().A[0], myMat3[:, 1].flatten().A[0], s=2, c='r')
    plt.show()

    myDat4 = loadDataSet('ex2test.txt')
    myMat4 = np.mat(myDat4)
    print '剪枝前树的深度; ', getTreeDepth(retTree3)
    print '剪枝前树的叶节点数; ', getNumLeafs(retTree3)
    retTree4 = prune(retTree3, myMat4)
    print retTree4
    print '剪枝后树的深度; ', getTreeDepth(retTree4)
    print '剪枝后树的叶节点数; ', getNumLeafs(retTree4)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5))  # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(myMat4[:, 0].flatten().A[0], myMat4[:, 1].flatten().A[0], s=2, c='r')
    plt.show()

def modelTreeTest():
    '''
    测试模型树
    :return:
    '''
    myMat = np.mat(loadDataSet('exp2.txt'))
    myTree1 = createTree(myMat, modelLeaf, modelErr, (1, 10))
    print myTree1
    print '树的深度; ', getTreeDepth(myTree1)
    print '树的叶节点数; ', getNumLeafs(myTree1)
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(myMat[:, 0].flatten().A[0], myMat[:, 1].flatten().A[0], s=2, c='r')
    # lx1 = np.arange(0.0, float(myTree1['spVal']), 0.001)
    # ws1 = myTree1['right']
    # lyHat1 = ws1[1] * lx1 + ws1[0]
    # ax.plot(lx1, np.mat(lyHat1).T.A)
    # lx2 = np.arange(float(myTree1['spVal']), 1.0, 0.001)
    # ws2 = myTree1['left']
    # lyHat2 = ws2[1] * lx2 + ws2[0]
    # ax.plot(lx2, np.mat(lyHat2).T.A)

    srtInd = myMat[:, 0].argsort(0) # 按照矩阵第1列从小到大进行排序, 返回排序后元素的索引值
    xSort = myMat[srtInd][:, 0, :]
    yHat = createForeCast(myTree1, myMat[:, 0], modelTreeEval)
    ax.plot(xSort[:, 0], yHat[srtInd][:, 0])

    plt.show()

def hybridTreeTest():
    '''
    树回归与线性回归的测试比较
    :return:
    '''
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    srtInd = testMat[:, 0].argsort(0) # 按照矩阵第1列从小到大进行排序, 返回排序后元素的索引值
    xSort = testMat[srtInd][:, 0, :]
    myTree1 = createTree(trainMat, ops=(1, 20)) # 生成回归树
    print myTree1
    print '树的深度; ', getTreeDepth(myTree1)
    print '树的叶节点数; ', getNumLeafs(myTree1)
    yHat1 = createForeCast(myTree1, testMat[:, 0])
    print np.corrcoef(yHat1, testMat[:, 1], rowvar=0)[0, 1] # 输出相关系数

    myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20)) # 生成模型树
    print myTree2
    print '树的深度; ', getTreeDepth(myTree2)
    print '树的叶节点数; ', getNumLeafs(myTree2)
    leaves = []
    getLeaves(myTree2, leaves)
    print '树的叶节点; ', leaves
    yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    print np.corrcoef(yHat2, testMat[:, 1], rowvar=0)[0, 1] # 输出相关系数

    ws, X, Y = linearSolve(trainMat) # 线性回归
    m = np.shape(testMat)[0]
    yHat3 = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print np.corrcoef(yHat3, testMat[:, 1], rowvar=0)[0, 1] # 输出相关系数

    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    ax = fig.add_subplot(111)
    ax.scatter(testMat[:, 0].flatten().A[0], testMat[:, 1].flatten().A[0], s=2, c='r')

    ax.plot(xSort[:, 0], yHat1[srtInd][:, 0], label=u'回归树')
    ax.plot(xSort[:, 0], yHat2[srtInd][:, 0], label=u'模型树')
    ax.plot(testMat[:, 0], yHat3, label=u'线性回归')
    plt.legend(loc='upper left')

    plt.show()

if __name__=='__main__':
    # testMat = np.mat(np.eye(4))
    # print testMat
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print mat0
    # print mat1
    # regTreeTest()
    # modelTreeTest()
    hybridTreeTest()