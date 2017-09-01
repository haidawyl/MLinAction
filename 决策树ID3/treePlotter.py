#!/usr/bin/python
#  -*- coding:utf-8 -*-
'''
使用文本注解绘制树节点
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8") # 创建决策节点字典
leafNode = dict(boxstyle="round4", fc="0.8") # 创建叶节点字典
arrow_args = dict(arrowstyle="<-") # 创建箭头字典

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    执行绘图功能
    :param nodeTxt:
    :param centerPt:
    :param parentPt:
    :param nodeType:
    :return:
    '''
    # createPlot.ax1: 定义全局变量绘图区(Python语言中所有的变量默认都是全局有效的)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    '''
    在父子节点间填充文本信息
    :param cntrPt:
    :param parentPt:
    :param txtString:
    :return:
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    '''
    绘制树
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    '''
    numLeafs = getNumLeafs(myTree) # 树的宽度
    depth = getTreeDepth(myTree) # 树的深度
    firstStr = myTree.keys()[0] # 树的根节点标签
    # 将x轴分成叶子节点数*2个相同的区域, 决策节点从当前区域向右循环移动叶子节点数+1个区域
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) # 绘制当前节点的文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode) # 绘制当前节点
    secondDict = myTree[firstStr] # 根节点的全部子节点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD # 为了绘制子节点将y轴的偏移量减小深度分之一(根节点在最上方)
    # 遍历根节点的全部子节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': # 判断节点的数据类型是否为字典
            plotTree(secondDict[key], cntrPt, str(key)) # 是则递归绘制树
        else: # 否则
            # 将x轴分成叶子节点数*2个相同的区域, 从当前区域向右循环移动2个区域作为新的当前区域, 同时也作为绘制叶子节点的区域
            # (1.0/plotTree.totalW = 2/2.0/plotTree.totalW)
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) # 绘制子节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key)) # 绘制子节点的文本信息
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD # 绘制完所有子节点以后, 恢复到原来y轴的偏移量

def createPlot():
    '''
    :return:
    '''
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题

    fig = plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    fig.clf() # 清空绘图区
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(u'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode) # 绘制决策节点
    plotNode(u'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode) # 绘制叶节点
    # plt.grid(True) # 显示网格
    plt.show()

def createPlot(inTree):
    '''
    :param inTree:
    :return:
    '''
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题

    fig = plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    fig.clf() # 清空绘图区
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree)) # 设置树的宽度
    plotTree.totalD = float(getTreeDepth(inTree)) # 设置树的深度
    # x轴, y轴的有效范围都是[0.0, 1.0]
    plotTree.xOff = -0.5 / plotTree.totalW # 已经绘制完成的节点的x轴偏移量
    plotTree.yOff = 1.0 # 已经绘制完成的节点的y轴偏移量, 默认是根节点的y轴偏移量
    plotTree(inTree, (0.5, 1.0), '') # 绘制树
    plt.show()

def getNumLeafs(myTree):
    '''
    获取叶节点的数目
    :param myTree:
    :return:
    '''
    numLeafs = 0 # 存储叶节点数目
    firstStr = myTree.keys()[0] # 树的根节点标签
    secondDict = myTree[firstStr] # 根节点的全部子节点
    # 遍历根节点的全部子节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': # 判断节点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key]) # 是则递归计算叶节点数量再+当前节点数目作为新的叶节点数目
        else: numLeafs += 1 # 否则叶节点数量+1
    return numLeafs # 返回叶节点数量

def getTreeDepth(myTree):
    '''
    获取树的层数, 根节点的层数为0
    :param myTree:
    :return:
    '''
    maxDepth = 0 # 存储最大层数
    firstStr = myTree.keys()[0] # 树的根节点标签
    secondDict = myTree[firstStr] # 根节点的全部子节点
    # 遍历根节点的全部子节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': # 判断节点的数据类型是否为字典
            thisDepth = 1 + getTreeDepth(secondDict[key]) # 是则递归计算树的层数再+1作为当前树的层数
        else: thisDepth = 1 # 否则当前树的层数为1
        if thisDepth > maxDepth: maxDepth = thisDepth # 如果当前树的层数大于最大层数, 则最大层数修改为当前树的层数
    return maxDepth # 返回树的最大层数

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

if __name__ == '__main__':
    # createPlot()

    myTree = retrieveTree(1)
    # myTree['no surfacing'][3] = 'maybe'
    print 'Leafs Number = ', getNumLeafs(myTree)
    print 'Tree Depth = ', getTreeDepth(myTree)
    createPlot(myTree)
