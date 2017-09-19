#!/usr/bin/python
#  -*- coding:utf-8 -*-

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue # 节点元素名称
        self.count = numOccur # 出现次数
        self.nodeLink = None # 指向下一个相似元素节点的指针, 默认为None
        self.parent = parentNode # 指向父节点的指针
        self.children = {} # 存储子节点的字典, 以子节点的元素名称为键, 指向子节点的指针为值, 初始化为空字典

    def inc(self, numOccur):
        '''
        对出现次数count增加给定值numOccur
        :param numOccur:
        :return:
        '''
        self.count += numOccur

    def disp(self, ind=1):
        '''
        以文本形式打印树
        :param ind:
        :return:
        '''
        print '  '*ind, self.name, '', self.count
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1):
    '''
    构建FP(Frequent Pattern, 频繁模式)树
    :param dataSet: 数据集, 它实际上是一个字典, 字典的键是frozenset(由原始数据集的一行映射而成), 值是1.
    :param minSup: 最小支持度阈值
    :return: FP树和头指针字典
    '''
    headerTable = {} # 头指针字典
    # 第一次遍历数据集统计每个元素项出现的频度并存储到头指针字典中
    for trans in dataSet:
        for item in trans:
            # 字典(Dict)的get()函数返回指定键的值, 如果值不存在则返回默认值.
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 遍历头指针字典删除那些出现次数少于minSup的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    # 将头指针字典的键转换成为set集合用于创建频繁项集, 此时头指针字典中仅包含满足最小支持度要求的元素项
    freqItemSet = set(headerTable.keys())
    # 如果频繁项集为空, 即所有元素项都不满足最小支持度要求, 则返回空值
    if len(freqItemSet) == 0:
        return None, None
    # 对头指针字典进行扩展以便能够保存计数值和指向每种类型第一个元素项的指针, 例如:
    # 转换前: {'s': 3}
    # 转换后: {'s': [3, None]}
    for k in headerTable:
        # 将头指针字典的值由原来的数字(出现次数)更改为列表[出现次数, None],
        # "None"将会在后面被更改为指向本类型第一个元素项的指针
        headerTable[k] = [headerTable[k], None]
    # 创建根节点为空集的FP树
    retTree = treeNode('Null Set', 1, None)
    # 第二次遍历数据集, 只针对频繁项进行处理. 字典(Dict)的items()函数返回可遍历的(键, 值)元组数组
    for tranSet, count in dataSet.items():
        localD = {} # 保存{元素, 次数}对
        # 查找数据集的当前行中有哪些元素项出现在频繁项集中, 将这些元素项及其在数据集中出现的次数存储到localD中
        for item in tranSet:
            if item in freqItemSet: # 如果元素项在频繁项集中存在
                localD[item] = headerTable[item][0] # 将元素项和头指针字典中元素项的出现次数存储到localD中
        if len(localD) > 0: # 当前行中存在满足最小支持度要求的元素项
            # 按照出现次数降序排序之后, 再创建排序后的元素项列表
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            # 更新FP树
            # orderedItems: 满足最小支持度要求的元素项列表
            # retTree: 树的根节点, 为空值
            # headerTable: 头指针字典
            # count: orderedItems中全部元素所在行的次数, 即为1
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable # 返回FP树和头指针字典

def updateTree(items, inTree, headerTable, count):
    '''
    更新FP(Frequent Pattern, 频繁模式)树
    按照items中元素的顺序递归创建树的子节点, 即items[0]是inTree的子节点, items[1]是items[0]的子节点,
    items[2]是items[1]的子节点, 依此类推.
    本函数是一个递归函数, 在本函数的每次调用中, 会不断地添加新的节点到FP树以及更新头指针字典的元素指针链表
    :param items: 按照出现次数降序排序的满足最小支持度要求的元素项列表, 所有元素都出现在原数据集的某一行中
    :param inTree: 父节点, 即在该节点下递归创建items中的所有子节点
    :param headerTable: 头指针字典
    :param count: items中全部元素所在行的次数, 即为1
    :return:
    '''
    # items是按照出现次数降序排序的元素项列表
    if items[0] in inTree.children: # 如果items的第一个元素项已经在inTree的子节点中, 则只需更新该元素项的计数值
        inTree.children[items[0]].inc(count) # 调用treeNode的inc()函数更新计数值
    else: # 否则, 创建一个新的treeNode并将其作为inTree的子节点添加到FP树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # 新建treeNode时指定父节点为inTree
        if headerTable[items[0]][1] == None: # 如果头指针字典中该元素项的指针为空, 则将其指针设置为该元素项在FP树中的节点treeNode对象
            headerTable[items[0]][1] = inTree.children[items[0]]
        else: # 如果头指针字典中该元素项的指针不为空, 则将该节点添加到头指针字典中该元素项的指针链表的末尾
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1: # 如果items的元素数目大于1, 则递归调用本函数, 在每次调用时去掉列表中的第一个元素, 参数inTree指定为items的第一个元素项在FP树中的节点对象
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    '''
    更新头指针字典, 确保节点链接指向树中该元素项的每一个节点实例
    在链表nodeToTest的末尾添加新的节点targetNode
    :param nodeToTest: 头指针字典中某元素项的头指针
    :param targetNode: 待追加的新节点
    :return:
    '''
    while (nodeToTest.nodeLink != None): # 从指定元素项的头指针开始, 遍历链表直到末尾.
        nodeToTest = nodeToTest.nodeLink
    # 将链表末尾节点的nodeLink指向新的节点targetNode, 即将targetNode添加到指针链表的末尾
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    '''
    将数据集转换为字典, 字典的键是数据集的行映射成的frozenset, 值是1.
    frozenset是指被"冰冻"的集合, 就是说它们是不可改变的.
    :param dataSet: 数据集
    :return:
    '''
    retDict = {}
    for trans in dataSet: # 遍历数据集, 将每一行映射成frozenset作为字典的键, 值设为1
        retDict[frozenset(trans)] = 1
    return retDict

def ascendTree(leafNode, prefixPath):
    '''
    查找以给定元素项结尾的前缀路径(递归查找leafNode的父节点)
    :param leafNode: 给定元素项
    :param prefixPath: 前缀路径
    :return:
    '''
    if leafNode.parent != None: # 给定元素项的父节点不为空
        prefixPath.append(leafNode.name) # 将给定元素项的名称加入到前缀路径
        ascendTree(leafNode.parent, prefixPath) # 使用给定元素项的父节点递归调用本方法

def findPrefixPath(basePat, treeNode):
    '''
    生成条件模式基
    :param basePat:
    :param treeNode: 头指针字典中某个元素项的指针链表的第一个节点
    :return: 条件模式基字典
    '''
    condPats = {} # 条件模式基字典
    # 指针链表保存的是相同的元素项, 每一个链表节点通过递归查找父节点可以找到其前缀路径
    # 根据这些链表节点的前缀路径和当前节点的元素项频率可以构建一个条件模式基
    while treeNode != None: # 遍历指针链表直到结尾
        prefixPath = [] # 前缀路径
        ascendTree(treeNode, prefixPath) # 调用函数ascendTree()查找以treeNode元素项结尾的前缀路径
        if len(prefixPath) > 1: # 如果前缀路径长度大于1
            # 将前缀路径除第一个元素以外的所有元素映射到frozenset,
            # 以该frozenset作为键, 当前元素项的频率作为值, 添加到条件模式基字典中
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink # 跳到指针链表的下一个节点继续生成条件模式基
    return condPats # 返回条件模式基字典

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    '''
    查找频繁项集
    举例来说, 假设t的条件模式基是{y,x,s,z}2, {y,x,r,z}1, 那么构建t的条件FP树的过程如下:
    1.递归0.
    (1).将[t]加入到频繁项集列表freqItemList中;
    (2).获取t的条件模式基是{y,x,s,z}2, {y,x,r,z}1;
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        生成的条件FP树T1为:
        Null Set  1
          y  3
            x  3
              z  3
        生成的头指针字典H1为{y, x, z}, 因为s和r不满足最小支持度要求, 所以删掉了.
    (4).使用T1,H1,minSup,[t],freqItemList递归调用本函数.

    2.递归1, 从1进入.
    (1).将[y,t]加入到频繁项集列表freqItemList中;
    (2).获取y的条件模式基是{};
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        条件FP树和头指针字典都为空, 不做任何处理.

    3.递归1, 从1进入.
    (1).将[x,t]加入到频繁项集列表freqItemList中;
    (2).获取x的条件模式基是{y}3;
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        生成的条件FP树T2为:
        Null Set  1
          y  3
        生成的头指针字典H2为{y}.
    (4).使用T2,H2,minSup,[x,t],freqItemList递归调用本函数.

    4.递归2, 从3进入.
    (1).将[y,x,t]加入到频繁项集列表freqItemList中;
    (2).获取y的条件模式基是{};
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        条件FP树和头指针字典都为空, 不做任何处理.

    5.递归1, 从1进入.
    (1).将[z,t]加入到频繁项集列表freqItemList中;
    (2).获取z的条件模式基是{y,x}3;
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        生成的条件FP树T3为:
        Null Set  1
          y  3
            x  3
        生成的头指针字典H3为{y,x}.
    (4).使用T3,H3,minSup,[z,t],freqItemList递归调用本函数.

    6.递归2, 从5进入.
    (1).将[x,z,t]加入到频繁项集列表freqItemList中;
    (2).获取x的条件模式基是{y}3;
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        生成的条件FP树T4为:
        Null Set  1
          y  3
        生成的头指针字典H4为{y}.
    (4).使用T4,H4,minSup,[x,z,t],freqItemList递归调用本函数.

    7.递归3, 从6进入.
    (1).将[y,x,z,t]加入到频繁项集列表freqItemList中;
    (2).获取y的条件模式基是{};
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        条件FP树和头指针字典都为空, 不做任何处理.

    8.递归2, 从5进入.
    (1).将[y,z,t]加入到频繁项集列表freqItemList中;
    (2).获取y的条件模式基是{};
    (3).使用条件模式基和最小支持度阈值生成条件FP树和头指针字典;
        条件FP树和头指针字典都为空, 不做任何处理.
    :param inTree: FP树
    :param headerTable: 头指针字典
    :param minSup: 最小支持度阈值
    :param preFix: 频繁项集前缀(集合)
    :param freqItemList: 频繁项集列表
    :return:
    '''
    # 对头指针字典中的元素项按照其出现的频率进行升序排序, 将排序后的键组合成频繁项列表, 头指针字典的键为单元素项.
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])]
    for basePat in bigL: # 遍历频繁项(单元素项)列表.
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat) #  将本次遍历得到的频繁项(单元素项)插入到频繁项集前缀(集合)的最前面生成新的频繁项集
        freqItemList.append(newFreqSet) # 将新生成的频繁项集添加到频繁项集列表中
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1]) # 使用该元素项的指针链表调用函数findPrefixPath()生成条件模式基
        myCondTree, myHead = createTree(condPattBases, minSup) # 使用条件模式基和最小支持度阈值生成条件FP树和新的头指针字典
        if myHead != None: # 如果新生成的条件FP树中有元素, 则递归调用本函数查找频繁项集
            print 'conditional tree for: ', newFreqSet
            myCondTree.disp(1)
            # myCondTree: 前面生成的条件FP树
            # myHead: 前面生成的头指针字典
            # minSup: 最小支持度阈值
            # newFreqSet: 本次生成的频繁项集, 作为迭代调用函数mineTree()的频繁项集前缀(集合)
            # freqItemList: 频繁项集列表
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

# import twitter
# from time import sleep
# import re
# def getLotsOfTweets(searchStr):
#     '''
#     根据关键词查询tweets
#     :param searchStr: 查询关键词
#     :return: 返回tweets
#     '''
#     CONSUMER_KEY = ''
#     CONSUMER_SECRET = ''
#     ACCESS_TOKEN_KEY = ''
#     ACCESS_TOKEN_SECRET = ''
#     api = twitter.Api(consumer_key=CONSUMER_KEY,
#                       consumer_secret=CONSUMER_SECRET,
#                       access_token_key=ACCESS_TOKEN_KEY,
#                       access_token_secret=ACCESS_TOKEN_SECRET)
#     # 获取14页每页100条推文.
#     resultsPages = []
#     for i in range(1, 15):
#         print 'fetching page %d' % i
#         searchResults = api.GetSearch(searchStr, per_page=100, page=i)
#         resultsPages.append(searchResults)
#         sleep(6)
#     return resultsPages
#
# def textParse(bigString):
#     '''
#     解析文本
#     :param bigString: 待解析的文本
#     :return:
#     '''
#     urlsRemoved = re.sub('(http[s]?:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString) # 去掉URL
#     listOfTokens = re.split(r'\W*', urlsRemoved) # 使用除单词和数字以外的任意字符串分隔文本
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2] # 返回解析后的长度大于2的单词, 并将大写字母转换为小写字母
#
# def mineTweets(tweetArr, minSup=5):
#     '''
#     构建FP树对推文内容进行挖掘.
#     :param tweetArr: 获取到的推文列表
#     :param minSup: 最小支持度阈值
#     :return: 频繁项集列表
#     '''
#     parsedList = []
#     for i in range(14):
#         for j in range(100):
#             parsedList.append(textParse(tweetArr[i][j].text)) # 调用函数textParse()解析推文
#     initSet = createInitSet(parsedList)
#     myFPtree, myHeaderTab = createTree(initSet, minSup)
#     myFreqList = []
#     mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
#     return myFreqList # 返回频繁项集列表

if __name__=='__main__':
    # rootNode = treeNode('pyramid', 9, None)
    # rootNode.children['eye'] = treeNode('eye', 13, None)
    # rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # rootNode.disp()

    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    for item in myHeaderTab:
        condPats = findPrefixPath(item, myHeaderTab[item][1])
        print item, ':', condPats

    freqItems = [] # 存储频繁项集
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print freqItems

    # lotsOtweets = getLotsOfTweets('RIMM')
    # listOfTerms = mineTweets(lotsOtweets, 20)

    # parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    # initSet = createInitSet(parsedDat)
    # myFPtree, myHeaderTab = createTree(initSet, 100000)
    # myFreqList = []
    # mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    # print myFreqList