#!/usr/bin/python
#  -*- coding:utf-8 -*-

import math
import operator
import treePlotter

def calcShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵
    :param dataSet: 计算香农熵的数据集
    :return:
    '''
    numEntries = len(dataSet) # 取得数据集的行数
    labelCounts = {} # 存储最后一列数值及其出现次数的字典
    # 遍历数据集
    for featVec in dataSet:
        currentLabel = featVec[-1] # 取得每行数据的最后一列
        if currentLabel not in labelCounts.keys(): # 如果当前行最后一列数值不在字典中, 则将其加入字典, 同时设置出现次数为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 # 将当前行最后一列数值在字典中的出现次数+1
    shannonEnt = 0.0 # 存储香农熵
    # 遍历字典
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries # 求字典中每一项出现的概率, 即数据集中最后一列数值出现的概率, p = 该数值出现的次数 / 总的数据量
        shannonEnt -= prob * math.log(prob, 2) # 计算香农熵
    return shannonEnt # 返回香农熵

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征值
    :return:
    '''
    retDataSet = [] # 创建新的列表对象, 用于存储返回的数据
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value: # 如果某数据在参数指定的特征上的值和参数传递过来的特征值相等
            reducedFeatVec = featVec[:axis] # 取得该特征值之前的所有特征值赋给新的特征向量
            reducedFeatVec.extend(featVec[axis+1:]) # 再将该特征值之后的所有特征值追加到新的特征向量的末尾
            # 执行上述2条语句的结果是, 将原数据中除满足条件的特征值以外的所有特征值赋给新的特征向量
            # append方法 和 extend方法的区别: 这2个方法处理多个列表的结果完全不同
            # 例: a = [1, 2, 3] 和 b = [4, 5, 6], 则
            # a.append(b) ==> [1, 2, 3, [4, 5, 6]], 使用append方法列表得到了第4个元素, 而且第4个元素也是一个列表;
            # a.extend(b) ==> [1, 2, 3, 4, 5, 6], 使用extend方法得到一个包含a和b所有元素的列表.
            retDataSet.append(reducedFeatVec) # 将新的特征向量追加到返回数据集中
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet: 待划分的数据集
    :return: 最好特征划分的index(索引)值
    '''
    numFeatures = len(dataSet[0]) - 1 # 取得数据集上的特征属性数量, 最后一列是该行实例数据的类别标签
    m = len(dataSet) # 数据集的行数
    baseEntropy = calcShannonEnt(dataSet) # 计算完整数据集的原始香农熵
    bestInfoGain = 0.0 # 存储最大的信息增益
    bestFeature = -1 # 存储最好的划分数据集的特征的index(索引)
    # 遍历数据集的全部特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 取得所有数据的第i个特征的属性值存储到列表中
        uniqueVals = set(featList) # 去除列表中的重复值
        newEntropy = 0.0 # 存储新的香农熵
        # 遍历当前特征中的所有唯一属性值
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # 对每个唯一属性值划分一次数据集
            # prob = len(subDataSet)/float(len(dataSet)) # 每个唯一属性值出现的概率 = 划分后新的数据集的行数 / 原始数据集的行数
            prob = len(subDataSet)/float(m) # 同上
            # 每个唯一属性值在原始数据集上的香农熵 = 该唯一属性值在原始数据集上出现的概率 * 该唯一属性值的数据子集的香农熵,
            # 然后对所有唯一属性值在原始数据集上的香农熵求和即得到第i个特征值在原始数据集上的香农熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益是香农熵的减少量
        infoGain = baseEntropy - newEntropy # 第i个特征值的信息增益 = 原始香农熵 - 第i个特征值在原始数据集上的香农熵
        if (infoGain > bestInfoGain): # 判断第i个特征值的信息增益是否大于当前最大的信息增益, 是则执行
            bestInfoGain = infoGain # 最大信息增益修改为第i个特征值在原始数据集上的信息增益
            bestFeature = i # 最好特征划分的index(索引)修改为i
    return bestFeature # 返回最好特征划分的index(索引)值

def majorityCnt(classList):
    '''
    本函数使用分类名称的列表, 然后创建键值为classList中唯一值的数据字典, 字典对象存储了classList
    中每个类标签出现的频率, 最后利用operator操作键值排序字典, 并返回出现次数最多的分类名称.
    :param classList: 分类名称列表
    :return: 返回出现次数最多的分类名称
    '''
    classCount = {} # 创建键值为classList中唯一值的数据字典
    # 遍历classList
    for vote in classList:
        if vote not in classCount.keys(): # 如果classList中的值不在classCount的键值中, 则加入到其中, 同时设置对应的出现次数为0
            classCount[vote] = 0
        classCount[vote] += 1 # 相同的键值其对应的出现次数+1
    # iteritems()返回字典项的迭代器
    # sorted(iterable, cmp=None, key=None, reverse=False):
    # iterable: 待排序的迭代器;
    # cmp: 比较函数;
    # key: 迭代器中的某个元素作为关键字, operator.itemgetter(index)返回迭代器中索引对应的元素;
    # reverse: 排序规则, reverse=True表示降序, reverse=False表示升序, 默认值为False.
    # 返回排序后的元祖列表
    # 按照出现次数降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # 返回出现次数最多的分类名称

def createTree(dataSet, labels):
    '''
    创建树的函数代码
    :param dataSet: 样本数据集
    :param labels: 包含数据集中所有特征的标签列表
    :return: 树的字典实例
    '''
    classList = [example[-1] for example in dataSet] # 创建包含样本数据集所有类标签的列表
    # List.count()方法用于统计某个元素在列表中出现的次数.
    # 如果列表中第1个元素出现的次数与该列表的长度相等, 即该列表中所有元素完全相同, 则直接返回该列表的第1个元素.
    # 换个角度理解, 当所有的类标签完全相同时, 则直接返回该类标签, 这是该递归函数的第1个停止条件.
    # 这时该叶子节点的所有元素都属于同一类.
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果样本数据集只剩下1列, 即类标签列, 这说明使用完了所有特征, 仍然不能将数据集划分成仅包含唯一类别的分组,
    # 则返回该数据集中出现次数最多的类标签, 这是该递归函数的第2个停止条件.
    # 这时该叶子节点的所有元素不属于同一类, 我们只需要将其标注为出现次数最多的类别即可.
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择划分当前数据集最好的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 取得当前所选特征的标签
    bestFeatLabel = labels[bestFeat]
    # 开始创建树
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) # 删除标签列表中所选特征对应的标签
    featValues = [example[bestFeat] for example in dataSet] # 创建包含所选特征的所有属性值的列表
    uniqueVals = set(featValues) # 去除列表中重复的属性值
    # 遍历当前所选特征包含的全部唯一属性值
    for value in uniqueVals:
        subLabels = labels[:] # 创建包含所选特征标签以外的全部标签的列表(函数参数是列表类型时, 参数按照引用方式传递)
        # 根据当前特征和该特征下的当前属性值划分数据集, 根据划分后新的数据集和新的标签列表创建子树,
        # 并将该子树指向当前特征下的当前属性值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree # 返回创建的树

def classify(inputTree, featLabels, testVec):
    '''
    使用决策树的分类函数
    :param inputTree: 决策树
    :param featLabels: 特征标签列表
    :param testVec: 测试向量
    :return:
    '''
    firstStr = inputTree.keys()[0] # 树的根节点标签
    secondDict = inputTree[firstStr] # 根节点的全部子节点
    featIndex = featLabels.index(firstStr) # 将标签字符串转换为索引
    # 遍历根节点的全部子节点
    for key in secondDict.keys():
        if testVec[featIndex] == key: # 比较测试向量的特征值和树节点的值
            # 若相等, 则说明测试向量属于这个节点或者其子节点
            if type(secondDict[key]).__name__ == 'dict': # 如果当前树节点非叶子节点, 则递归调用该决策树的分类函数, 直至找到测试向量所属的类别
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key] # 如果当前树节点是叶子节点, 则测试向量属于这个节点的类别
    return classLabel # 返回节点的分类标签

def storeTree(inputTree, filename):
    '''
    存储决策树
    :param inputTree:
    :param filename:
    :return:
    '''
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '''
    加载决策树
    :param filename:
    :return:
    '''
    import pickle
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    # dataSet, labels = createDataSet()
    # print 'dataSet = \n', dataSet
    # print 'labels = \n', labels

    # shannonEnt = calcShannonEnt(dataSet)
    # print 'shannonEnt = ', shannonEnt
    # dataSet[0][-1] = 'maybe'
    # print 'dataSet = \n', dataSet
    # shannonEnt = calcShannonEnt(dataSet)
    # print 'shannonEnt = ', shannonEnt # 熵越高, 则混合的数据越多

    # print 'dataSet = \n', dataSet
    # newDataSet01 = splitDataSet(dataSet, 0, 1)
    # print 'newDataSet = \n', newDataSet01
    # newDataSet00 = splitDataSet(dataSet, 0, 0)
    # print 'newDataSet = \n', newDataSet00

    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print 'bestFeature = ', bestFeature

    # copyLabels = labels[:]
    # myTree = createTree(dataSet, copyLabels)
    # print myTree
    #
    # classLabel = classify(myTree, labels, [1, 0])
    # print 'classLabel = ', classLabel
    # classLabel = classify(myTree, labels, [1, 1])
    # print 'classLabel = ', classLabel
    #
    # storeTree(myTree, 'classifierStorage.txt')
    # print grabTree('classifierStorage.txt')

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRage']
    lensesTree = createTree(lenses, lensesLabels)
    print lensesTree
    treePlotter.createPlot(lensesTree)
