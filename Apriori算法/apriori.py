#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np

'''
 Aprioir算法:
 (1). 构建集合C1, 包含数据集D中所有不重复的单元素生成的集合
 (2). 扫描数据集D找到C1中满足最小支持度要求的项集, 生成集合L1
 (3). 集合L1中的元素相互组合构成集合C2, C2中项集元素的数目等于L1中项集元素的数目+1
 (4). 扫描数据集D找到C2中满足最小支持度要求的项集, 生成集合L2
 (5). (3)中由L(k-1)生成Ck, (4)中由Ck生成Lk同时k+1, 不断循环(3)和(4)两个步骤直到Lk为空
 支持度=项集在数据集D中出现的次数/数据集的总数, 如果项集的元素数目大于1, 则项集在数据集D
 的某条记录中是否出现的判断标准是所有元素在该记录中是否同时出现.
 一条规则可以表示成P->H(P和H都是集合), 其中集合P称为前件, 集合H称为后件, 规则P->H的
 可信度=support(P|H)/support(P), support即为前面计算得到的支持度, P|H是集合P和集合H
 的并集. 前面循环生成的L1, L2, ...Lk中的项集实际上就是P|H, H初始值是项集的单个元素组成的
 集合, 然后将这些单元素依次合并生成包含2个, 3个, ...n个(n为项集单元素的数目)元素的集合作为
 新的H, 再利用这些P|H和H计算可信度, 最后找到满足最小可信度要求的规则.
'''

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    '''
    创建数据集中所有不重复的单元素项集
    :param dataSet: 数据集
    :return:
    '''
    C1 = [] # 存储所有不重复的单元素项集
    for transaction in dataSet: # 遍历数据集
        for item in transaction: # 遍历数据集的每一条记录
            # 如果某个物品项没有在C1中, 则将其添加到C1中. 这里并不是简单地添加每个物品项, 而是
            # 添加只包含该物品项的一个列表. 这样做的目的是为每个物品项构建一个集合. 因为在
            # Apriori算法的后续处理中, 需要做集合操作.
            if not [item] in C1:
                C1.append([item])
    C1.sort() # 对C1进行排序
    # 将C1中每个单元素列表映射到frozenset, frozenset是指被"冰冻"的集合, 就是说它们是不可改变的.
    return map(frozenset, C1)

def scanD(D, Ck, minSupport):
    '''
    从Ck生成Lk
    :param D: 数据集
    :param Ck: 候选项集集合Ck
    :param minSupport: 最小支持度
    :return: 满足最小支持度要求的项集集合和所有项集及其支持度的字典
    '''
    ssCnt = {}
    # 计算Ck中每个候选项集在数据集中出现的次数, 候选项集中包含多个元素, 则计算所有元素同时出现的次数
    # 项集在数据集的某条记录中是否出现的判断标准是所有元素在该记录中是否同时出现
    for tid in D: # 遍历数据集
        for can in Ck: # 遍历Ck中的候选项集
            # 如果Ck中的集合是记录的一部分, 那么增加字典中对应的计数值. 这里字典的键就是项集的集合
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D)) # 数据集的总数目
    retList = [] # 包含满足最小支持度要求的项集集合
    supportData = {} # 包含项集及其支持度的字典
    # 计算每个项集的支持度, 即每个项集在数据集中出现的次数除以数据集的总数,
    # 同时判断该支持度是否小于给定的最小支持度, 是则丢弃掉, 否则保留
    for key in ssCnt:
        support = ssCnt[key] / numItems # 计算支持度, 项集在数据集中出现的次数除以数据集的总数
        if support >= minSupport: # 如果支持度大于最小支持度
            retList.insert(0, key) # 在列表的首部插入该项集
        supportData[key] = support # 设置项集的支持度
    return retList, supportData # 返回满足最小支持度要求的项集集合和所有项集及其支持度的字典

def aprioriGen(Lk, k):
    '''
    创建元素数目为k的候选项集集合Ck
    将频繁项集集合Lk(实际上是前一个频繁项集集合L(k-1))中的元素以每k个为一组生成新的非重复集合.例如:
    L(k-1) = [frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])] 且 k = 2 则
    Ck = [frozenset([1, 3]), frozenset([1, 2]), frozenset([1, 5]), frozenset([2, 3]), frozenset([3, 5]), frozenset([2, 5])]
    L(k-1) = [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])] 且 k = 3 则
    Ck = [frozenset([2, 3, 5])]
    :param Lk: 频繁项集集合Lk, 实际上是前一个频繁项集集合L(k-1), 每个项集中的元素数目为k-1
    :param k: 项集元素数目k
    :return: 候选项集集合Ck
    '''
    retList = [] # 候选项集集合
    lenLk = len(Lk) # 频繁项集集合Lk中项集的数目
    # 当利用{0},{1},{2}构建{0,1},{0,2},{1,2}时, 实际上是将单个项组合到一块. 如果想利用{0,1},{0,2},{1,2}
    # 来创建三元素项集, 应该怎么做? 如果将每两个集合合并, 就会得到{0,1,2},{0,1,2},{0,1,2}, 也就是说,
    # 同样的结果集合会重复3次. 接下来需要扫描三元素项集列表来得到非重复结果, 我们要做的是确保遍历列表的
    # 次数最少. 如果比较集合{0,1},{0,2},{1,2}的第1(k-2)个元素并只对第1个元素相同的集合求并操作, 就得到
    # 我们想要的结果{0,1,2}了, 而且只有一次操作. 同时也不需要遍历列表来寻找非重复值了.
    for i in range(lenLk): # 遍历索引为0到lenLk-1的频繁项集集合Lk
        for j in range(i+1, lenLk): # 遍历索引为i+1到lenLk-1的频繁项集集合Lk
            L1 = list(Lk[i])[:k-2] # 列表Lk[i]的前k-2项
            L2 = list(Lk[j])[:k-2] # 列表Lk[j]的前k-2项
            L1.sort() # 排序
            L2.sort() # 排序
            if L1 == L2: # 列表Lk[i]和列表Lk[j]的前k-2项相同, 则合并列表Lk[i]和列表Lk[j]
                # |: 集合并操作
                retList.append(Lk[i] | Lk[j])
    return retList # 返回候选项集集合

def apriori(dataSet, minSupport=0.5):
    '''
    根据数据集dataSet生成满足最小支持度minSupport要求的频繁项集集合
    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return: 频繁项集集合及项集支持度字典
    '''
    C1 = createC1(dataSet) # 创建候选项集集合C1
    print 'C1 =', C1
    D = map(set, dataSet) # 将数据集转化为集合列表D
    L1, supportData = scanD(D, C1, minSupport) # 根据候选项集集合C1生成频繁项集集合L1
    print 'L1 =', L1
    L = [L1] # 将L1放入列表L中, L会包含L1,L2,L3...
    k = 2
    while(len(L[k-2])>0): # 前一个频繁项集集合不为空时, 根据它生成下一个候选项集集合
        Ck = aprioriGen(L[k-2], k) # 根据频繁项集集合L(k-1)生成候选项集集合Ck
        print 'C%d = %s' % (k, Ck)
        Lk, supK = scanD(D, Ck, minSupport) # 根据候选项集集合Ck生成频繁项集集合Lk, 在此过程中将丢掉不满足最小支持度要求的项集
        print 'L%d = %s' % (k, Lk)
        supportData.update(supK) # 更新项集支持度字典
        L.append(Lk) # 将Lk放入列表L中
        k += 1 # 增加k值
    return L, supportData # 返回频繁项集集合及项集支持度字典

def generateRules(L, supportData, minConf=0.7):
    '''
    生成满足最小可信度要求的规则列表
    :param L: 频繁项集列表
    :param supportData: 包含频繁项集支持度的字典
    :param minConf: 最小可信度阈值
    :return: 满足最小可信度要求的规则列表
    '''
    bigRuleList = [] # 满足最小可信度要求的规则集合
    for i in range(1, len(L)): # 遍历频繁项集列表, 只获取有两个或更多元素的集合
        for freqSet in L[i]: # 遍历频繁项集的元素, 该元素也是一个列表
            # 创建只包含频繁项集单个元素集合的列表, 也就是说列表的每个元素都是频繁项集单个元素的集合
            H1 = [frozenset([item]) for item in freqSet]
            # i为频繁项集列表L中每个频繁项集的索引, 同时(索引+1)也等于频繁项集中元素的个数.
            if (i > 1): # 如果频繁项集元素的数目超过2, 则尝试对项集进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: # 如果项集中只有两个元素, 则计算规则的可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList # 返回满足最小可信度要求的规则列表

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    计算规则的可信度并找到满足最小可信度要求的所有规则
    一条规则可以表示成P->H(P和H都是集合), 其中集合P称为前件, 集合H称为后件
    规则P->H的可信度 = support(P|H) / support(P)
    :param freqSet: 频繁项集, 实际上是P|H, 即集合P和集合H的并集
    :param H: 后件H的集合, 因为freqSet是集合P和集合H的并集, 所以集合P=freqSet-H
            如果每一个后件H是单个元素的集合时, 则这些元素来自于频繁项集freqSet,
            如果每一个后件H不是单个元素的集合时, 则这些后件是使用函数rulesFromConseq()
            合并得到的.
    :param supportData: 包含频繁项集支持度的字典
    :param brl: 规则可信度列表
    :param minConf: 最小可信度阈值
    :return: 满足最小可信度要求的规则的后件集合
    '''
    prunedH = [] # 保存所有满足最小可信度要求的规则的后件
    for conseq in H: # 遍历所有的后件
        # 计算规则的可信度, 规则可以表示成P->H, 则P->H的可信度 = support(P|H) / support(P)
        # P|H是集合P和集合H的并集, 也就是参数freqSet, conseq是每一个可能的后件H, 则
        # 前件P = (P|H) - H = freqSet - conseq, 那么规则P->H的可信度 =
        # support(freqSet) / support(freqSet - conseq)
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf: # 判断规则的可信度是否大于等于最小可信度
            # print freqSet-conseq, '-->', conseq, 'conf:', conf
            # 将满足最小可信度要求的规则加入到规则可信度列表brl中
            brl.append((freqSet-conseq, conseq, conf))
            # 保存满足最小可信度要求的规则的后件
            prunedH.append(conseq)
    return prunedH # 返回满足最小可信度要求的规则的后件集合, 用于函数rulesFromConseq()的递归生成候选规则

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    生成候选规则集合
    一条规则可以表示成P->H(P和H都是集合), 其中集合P称为前件, 集合H称为后件
    规则P->H的可信度 = support(P|H) / support(P)
    :param freqSet: 频繁项集, 实际上是P|H, 即集合P和集合H的并集
    :param H: 后件H的集合, 因为freqSet是集合P和集合H的并集, 所以集合P=freqSet-H
            如果每一个后件H是单个元素的集合时, 则这些元素来自于频繁项集freqSet,
            如果每一个后件H不是单个元素的集合时, 则这些后件来自于使用函数calcConf()
            得到的满足最小可信度要求的规则的后件, 由本函数的递归调用而来
    :param supportData: 包含频繁项集支持度的字典
    :param brl: 规则可信度列表
    :param minConf: 最小可信度阈值
    :return:
    '''
    m = len(H[0]) # 后件的元素个数
    if (len(freqSet) > (m + 1)): # 频繁项集的元素个数大于后件的元素个数+1时, 则需要合并后件中的元素生成数目为m+1的新后件
        Hmpl = aprioriGen(H, m + 1) # 创建元素数目为m+1的新后件集合
        Hmpl = calcConf(freqSet, Hmpl, supportData, brl, minConf) # 计算规则的可信度并返回满足最小可信度要求的规则的后件集合
        if (len(Hmpl) > 1): # 如果前面返回的后件集合数目大于1, 则递归调用本函数生成候选规则集合
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)

def pntRules(ruleList, itemMeaning):
    '''
    遍历规则列表输出规则的含义
    :param ruleList: 规则列表
    :param itemMeaning: 含义列表
    :return:
    '''
    for ruleTup in ruleList: # 遍历规则列表输出规则的含义
        for item in ruleTup[0]:
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]:
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2]
        print

from time import sleep
from votesmart import votesmart

votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# votesmart.apikey = 'get your api key first'
def getActionIds():
    '''
    :return: 行为ID列表和议案标题列表
    '''
    actionIdList = [] # 行为ID列表
    billTitleList = [] # 议案标题列表
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) # 调用API获得一个billDetail对象
            for action in billDetail.actions: # 遍历议案中的所有行为
                # 过滤出包含投票的行为
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1) # 休眠1秒
    return actionIdList, billTitleList # 返回行为ID列表和议案标题列表

def getTransList(actionIdList, billTitleList):
    '''
    :param actionIdList: 行为ID列表
    :param billTitleList: 议案标题列表
    :return: 政客的政党信息和对议案的投票结果, 含义列表
    '''
    # 创建含义列表, 当想知道某些元素项的具体含义时, 使用元素项的值作为索引访问itemMeaning即可
    itemMeaning = ['Republican', 'Democratic']
    for billTitle in billTitleList: # 遍历议案标题列表
        itemMeaning.append('%s -- Nay' % billTitle) # 在议案标题后添加" -- Nay"(反对)
        itemMeaning.append('%s -- Yea' % billTitle) # 在议案标题后添加" -- Yea"(同意)
    # 议案元素项字典, 键值是政客的名字, 值的第一项是他的政党信息, 后面的项是他对议案的投票结果
    # 投票结果是个数字, 该数字作为索引访问itemMeaning即可知道它的具体含义.
    transDict = {}
    voteCount = 2
    for actionId in actionIdList: # 遍历行为ID列表
        sleep(3) # 休眠3秒
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId) # 调用API获取某个特定actionId相关的所有投票信息.
            for vote in voteList: # 遍历所有的投票信息
                if not transDict.has_key(vote.candidateName): # 某个政客的名字在字典中不存在
                    # 字典中的每个政客都有一个列表来存储他的政党信息和他投票的元素项
                    transDict[vote.candidateName] = [] # 政客的名字作为字典的键值
                    # 添加政党信息
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1) # 共和党
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0) # 民主党
                if vote.action == 'Nay': # 投反对票
                    transDict[vote.candidateName].append(voteCount) # 该值作为索引访问itemMeaning即可知道它的具体含义
                elif vote.action == 'Yea': # 投赞成票
                    transDict[vote.candidateName].append(voteCount + 1) # 该值作为索引访问itemMeaning即可知道它的具体含义
        except:
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning # 返回政客的政党信息和对议案的投票结果, 含义列表

def mushroomTest():
    '''
    :return:
    '''
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport=0.3)
    for i in range(len(L)):
        for item in L[i]:
            # a.intersection(b): 求集合a和b的交集
            if item.intersection('2'):
                print '包含2的项集: ', item
    rules = generateRules(L, suppData, minConf=0.99)
    for ruleTup in rules:
        for item in ruleTup[0]:
            print item,
        print "\t-------->\t",
        for item in ruleTup[1]:
            print item,
        print "confidence: %f" % ruleTup[2],
        print
        # print 'rule: ', ruleTup[0], '-->', ruleTup[1], ', 可信度: ', ruleTup[2]

if __name__=='__main__':
    dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # print C1
    # D = map(set, dataSet)
    # print D
    # L1, supportData = scanD(D, C1, 0.5)
    # print L1
    # print supportData

    # L, supportData = apriori(dataSet, minSupport=0.5)
    # print L
    # print supportData
    # rules = generateRules(L, supportData, minConf=0.5)
    # print rules

    # actionIdList, billTitleList = getActionIds()
    # transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    # dataSet = [transDict[key] for key in transDict.keys()]
    # L, suppData = apriori(dataSet, minSupport=0.3)
    # ruleList = generateRules(L, suppData, minConf=0.95)
    # pntRules(ruleList, itemMeaning)

    mushroomTest()