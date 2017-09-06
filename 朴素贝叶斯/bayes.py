#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import random
import feedparser
import operator

def loadDataSet():
    '''
    :return:
    '''
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], # class=0
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], # class=1
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], # class=0
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'], # class=1
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], # class=0
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] # class=1
    ]
    classVec = [0, 1, 0, 1, 0, 1] # 1:代表侮辱性文字, 0:代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    '''
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 文档集合
    :return:
    '''
    # set()集合操作符:
    # -: 差集
    # &: 交集
    # |: 合集 并集
    vocabSet = set([]) # 创建一个空的词汇表集合
    for document in dataSet: # 遍历所有文档
        vocabSet = vocabSet | set(document) # 将每篇文档的词汇集合添加到词汇表集合中, 求两个集合的并集并去重
    return list(vocabSet) # 返回词汇表

def setOfWords2Vec(vocabList, inputSet):
    '''
    词集模型(set-of-words model)
    将文档转换为词汇表向量
    :param vocabList: 词汇表集合
    :param inputSet: 某个文档
    :return: 文档向量, 向量的每个元素为1或0, 分别表示词汇表中的单词在输入文档中是否出现
    '''
    returnVec = [0]*len(vocabList) # 创建一个以0填充的和词汇表等长的向量
    for word in inputSet: # 遍历文档中的所有单词
        if word in vocabList: # 如果出现了词汇表中的单词
            returnVec[vocabList.index(word)] = 1 # 则将输出的文档向量中的对应值设为1
        else:
            print 'the word: %s is not in my Vocabulary!' % word
    return returnVec # 词集模型词汇表向量

def bagOfWords2VecMN(vocabList, inputSet):
    '''
    词袋模型(bag-of-words model)
    将文档转换为词汇表向量
    :param vocabList: 词汇表集合
    :param inputSet: 某个文档
    :return: 文档向量, 向量的每个元素分别表示词汇表中的单词在输入文档中出现的次数
    '''
    returnVec = [0]*len(vocabList) # 创建一个以0填充的和词汇表等长的向量
    for word in inputSet: # 遍历文档中的所有单词
        if word in vocabList: # 如果出现了词汇表中的单词
            returnVec[vocabList.index(word)] += 1 # 则将输出的文档向量中的对应值+1
    return returnVec # 词袋模型词汇表向量

def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    p(ci|w) = p(w|ci) * p(ci) / p(w)
    根据朴素贝叶斯的特征之间相互独立的假设得到; p(w|ci) = p(w0|ci)p(w1|ci)p(w2|ci)...p(wn|ci)
    :param trainMatrix: 文档矩阵
    :param trainCategory: 文档类别标签向量
    :return: p(w|c0), p(w|c1), p(c1), 另外p(c0) = 1 - p(c1)(这是一个二类分类问题)
    '''
    numTrainDocs = len(trainMatrix) # 文档矩阵的行数, 即文档的数量
    # 矩阵每一行是词汇表中单词在文档中存在与否的向量, 长度为词汇表的长度, 使用函数setOfWords2Vec()转换之后的结果
    numWords = len(trainMatrix[0]) # 文档矩阵的列数, 即词汇表的长度
    # sum(list)=list中各元素之和, 因为类别标签list中元素的值为0或1, 其加和得到类别标签为1的数量,
    # 再除以文档数量(等于类别标签list中元素的数量), 得到的是类别标签为1的概率, 即p(c1),
    # 由于这是一个二类分类问题, 所以p(c0) = 1 - p(c1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建以0填充的长度为numWords的NumPy数组p0Num, 保存类别标签为0的所有list相加之和,
    # 该list的每个元素表示这个元素对应词汇表中的单词在所有类别标签为0的文档中出现的次数
    # p0Num = np.zeros(numWords)
    # 利用贝叶斯分类器对文档进行分类时, 要计算多个概率的乘积以获得文档属于某个类的概率,
    # 即计算p(w0|ci)p(w1|ci)p(w2|ci)...p(wn|ci). 如果其中一个概率值为0, 则最后的的乘积也为0.
    # 为降低这种影响, 可以将所有词的出现数初始化为1, 并将分母初始化为2.
    p0Num = np.ones(numWords)
    # 创建以0填充的长度为numWords的NumPy数组p1Num, 保存类别标签为1的所有list相加之和,
    # 该list的每个元素表示这个元素对应词汇表中的单词在所有类别标签为1的文档中出现的次数
    # p1Num = np.zeros(numWords)
    p1Num = np.ones(numWords)
    # p0Denom = 0.0 # 保存类别标签为0的文档中所有单词的总数量
    p0Denom = 2.0
    # p1Denom = 0.0 # 保存类别标签为1的文档中所有单词的总数量
    p1Denom = 2.0
    for i in range(numTrainDocs): # 遍历全部文档
        if trainCategory[i] == 1: # 文档类别标签为1
            p1Num += trainMatrix[i] # list相加, 等于list中对应元素相加(sum(w0|c1), sum(w1|c1), sum(w2|c1), ..., sum(wn|c1))
            p1Denom += sum(trainMatrix[i]) # 求list中所有元素之和, 因元素值为0或1, 所以加和得到元素值为1的数量, 出现的单词总数
        else: # 文档类别标签为0
            p0Num += trainMatrix[i] # sum(w0|c0), sum(w1|c0), sum(w2|c0), ..., sum(wn|c0)
            p0Denom += sum(trainMatrix[i])
    # list/数字=list中各元素分别/该数字
    # p(w|c1)
    # p1Vect = p1Num / p1Denom # 运算结果表示类别标签为1的文档中各个单词在所有单词中的概率, (p(w0|c1), p(w1|c1), p(w2|c1), ..., p(wn|c1))
    # 另一个遇到的问题是下溢出, 这是由于太多很小的数相乘造成的. 当计算乘积p(w0|ci)p(w1|ci)p(w2|ci)...p(wn|ci)时,
    # 由于大部分因子都非常小, 所以程序会下溢出或者得到不正确的答案. 一种解决办法是对乘积取自然对数.
    # 在代数中有ln(a*b)=ln(a)+ln(b), 于是通过求对数可以避免下溢出或者浮点数舍入导致的错误.
    # 同时, 采用自然对数进行处理不会有任何损失.
    p1Vect = np.log(p1Num / p1Denom)
    # p(w|c0)
    # p0Vect = p0Num / p0Denom # 运算结果表示类别标签为0的文档中各个单词在所有单词中的概率, (p(w0|c0), p(w1|c0), p(w2|c0), ..., p(wn|c0))
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive # 返回p(w|c0)的自然对数, p(w|c1)的自然对数, p(c1)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    朴素贝叶斯分类函数
    :param vec2Classify: 词汇表中的单词在待分类的文档中存在与否的向量, 1xn向量, 长度为词汇表的长度
    :param p0Vec: p(w|c0), 即类别标签为0的文档中各个单词在所有单词中的概率(此处是概率值的自然对数), 1xn向量, 长度为词汇表的长度
    :param p1Vec: p(w|c1), 即类别标签为1的文档中各个单词在所有单词中的概率(此处是概率值的自然对数), 1xn向量, 长度为词汇表的长度
    :param pClass1: 类别标签为1的文档的概率
    :return:
    '''
    # p(ci|w) = p(w|ci) * p(ci) / p(w) => ln(p(ci|w)) = ln(p(w|ci) * p(ci) / p(w)) = ln(p(w|ci)) + ln(p(ci)) - ln(p(w))
    # sum(vec2Classify * p1Vec): 两个向量对应元素相乘, 得到的结果表示待分类的文档中出现的单词在类别1中各自的概率
    # 再将乘积得到的向量各个元素的概率相加求和, 得到的结果表示待分类的文档中出现的单词在类别1中的总概率, 即待分类的文档属于类别1的概率
    # 根据朴素贝叶斯的特征之间相互独立的假设得到; p(w) = p(w0)p(w1)p(w2)...p(wn)
    # 因为w是所有样本文档中出现的单词的集合, 也就是词汇表. 每个单词作为一个特征, 其概率就是1, 即p(wi)=1, 所以他们的乘积也为1(此种解释仔细推敲好像不对)
    # 根据p(ci|w) = p(w|ci) * p(ci) / p(w), 因为所有的概率都会除以p(w), 所以比较不同概率的大小等价于仅比较分子即p(w|ci) * p(ci)的大小
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1) # 待分类的文档属于类别1的概率
    # 待分类的文档属于类别0的概率, 因为这是一个二类分类问题, 所以1.0-pClass1等于类别标签为0的文档的概率
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0: # 属于类别1的概率大于属于类别0的概率则待分类的文档的类别为1
        return 1
    else: # 否则类别为0
        return 0

def textParse(bigString):
    '''
    文本解析
    :param bigString: 待解析的文本
    :return:
    '''
    import re
    listOfTokens = re.split(r'\W*', bigString) # 使用除单词和数字以外的任意字符串分隔文本
    wordList = [tok.lower() for tok in listOfTokens if len(tok) > 2] # 返回解析后的长度大于2的单词, 并将大写字母转换为小写字母

    # 移除停止词
    words = open('stopwords.txt').readlines()
    for word in words:
        word = word.strip().lower()
        while word in wordList:
            wordList.remove(word)

    return wordList

def testingNB():
    '''
    :return:
    '''
    listOPosts, listClasses = loadDataSet() # 文档列表和文档对应的分类列表
    myVocabList = createVocabList(listOPosts) # 生成字典表
    trainMat = [] # 训练矩阵
    for postinDoc in listOPosts: # 遍历文档列表
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc)) # 转换为词汇向量并加入训练矩阵
    p0V, p1V, pAb = trainNB0(trainMat, listClasses) # 训练样本文档并返回相应的概率值
    testEntry = ['love', 'my', 'dalmation'] # 测试文档
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry)) # 测试文档转换为词向量
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb) # 预测分类
    testEntry = ['stupid', 'garbage'] # 测试文档
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry)) # 测试文档转换为词向量
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb) # 预测分类

def spamTest():
    '''
    垃圾邮件分类测试
    :return:
    '''
    docList = [] # 文档列表
    classList = [] # 分类列表
    fullText = [] # 全部词条列表
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read()) # 读取垃圾邮件文件并解析该文件内容
        docList.append(wordList) # 将单词列表作为一个元素加入到docList中
        fullText.extend(wordList) # 将单词列表的每一个元素加入到fullText中
        classList.append(1) # 将分类列表中该文档的类别标签置为1即垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read()) # 读取非垃圾邮件文件并解析该文件内容
        docList.append(wordList) # 将单词列表作为一个元素加入到docList中
        fullText.extend(wordList) # 将单词列表的每一个元素加入到fullText中
        classList.append(0) # 将分类列表中该文档的类别标签置为0即非垃圾邮件
    vocabList = createVocabList(docList) # 生成词汇表
    trainingSet = range(50) # 整数列表, 值是0,1,2,...,49
    testSet = [] # 测试样本集
    # 随机选择10个文件作为测试样本. 这种随机选择数据的一部分作为训练集, 而剩余部分作为测试集的过程
    # 称为留存交叉验证(hold-out cross validation)
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet))) # 随机生成一个[0, 50)范围内的实数, 并转换为整数
        testSet.append(trainingSet[randIndex]) # 从训练样本集中随机选择一个样本放入测试样本集中
        del(trainingSet[randIndex]) # 从训练样本集中删除该样本
    trainMat = [] # 训练文档矩阵
    trainClasses = [] # 训练文档标签向量
    for docIndex in trainingSet: # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex])) # 对每封邮件基于词汇表并使用setOfWords2Vec()函数来构建词向量
        trainClasses.append(classList[docIndex]) # 每封邮件对应的类别标签
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses)) # 计算分类所需的概率
    errorCount = 0 # 分类错误数
    for docIndex in testSet: # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex]) # 对每封邮件基于词汇表并使用setOfWords2Vec()函数来构建词向量
        predictClass = classifyNB(np.array(wordVector), p0V, p1V, pSpam) # 对每封邮件预测分类
        if predictClass != classList[docIndex]: # 预测分类与真实分类进行比较
            errorCount += 1 # 如果分类错误则错误数+1
            print docList[docIndex]
            print '预测分类是', predictClass, ', 但实际分类却是', classList[docIndex]
    print 'the error rate is: ', float(errorCount)/len(testSet)

def calcMostFreq(vocabList, fullText):
    '''
    统计单词的频率, 返回频率最高的前N个单词
    :param vocabList: 词汇表
    :param fullText: 全文本
    :return: 出现频率最高的前30个单词
    '''
    freDict = {} # 词汇频率字典
    for token in vocabList: # 遍历词汇表
        freDict[token] = fullText.count(token) # 统计每个词在全文本中出现的次数, 并加入到词汇频率字典中
    sortedFreq = sorted(freDict.iteritems(), key=operator.itemgetter(1), reverse=True) # 按照出现次数从高到底对词汇频率字典进行排序
    return sortedFreq[:30] # 返回出现频率最高的前30个单词

def localWords(feed1, feed0):
    '''
    获取区域相关的词汇
    :param feed1:
    :param feed0:
    :return: 词汇表, 类别为0的概率, 类别为1的概率
    '''
    docList = [] # 文档列表
    classList = [] # 分类列表
    fullText = [] # 全部词条列表
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList) # 将单词列表作为一个元素加入到docList中
        fullText.extend(wordList) # 将单词列表的每一个元素加入到fullText中
        classList.append(1) # 将分类列表中该文档的类别标签置为1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList) # 将单词列表作为一个元素加入到docList中
        fullText.extend(wordList) # 将单词列表的每一个元素加入到fullText中
        classList.append(0) # 将分类列表中该文档的类别标签置为1
    vocabList = createVocabList(docList) # 生成词汇表
    top30Words = calcMostFreq(vocabList, fullText) # 获取出现频率最高的前30个单词
    print 'Top 30 Words: ', top30Words
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0]) # 从词汇表中去掉出现频率最高的那些词
    trainingSet = range(2*minLen) # 训练集索引列表
    testSet = [] # 测试集
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet))) # 随机生成一个[0, 2*minLen)范围内的实数, 并转换为整数
        testSet.append(trainingSet[randIndex]) # 从训练样本集中随机选择一个样本放入测试样本集中
        del(trainingSet[randIndex]) # 从训练样本集中删除该样本
    trainMat = [] # 训练文档矩阵
    trainClasses = [] # 训练文档标签向量
    for docIndex in trainingSet: # 遍历训练集
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) # 对每个文档基于词汇表并使用bagOfWords2VecMN()函数来构建词向量
        trainClasses.append(classList[docIndex]) # 每个文档对应的类别标签
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses)) # 计算分类所需的概率
    errorCount = 0 # 分类错误数
    for docIndex in testSet: # 遍历测试集
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex]) # 对每个文档基于词汇表并使用bagOfWords2VecMN()函数来构建词向量
        predictClass = classifyNB(np.array(wordVector), p0V, p1V, pSpam) # 对每个文档预测分类
        if predictClass != classList[docIndex]: # 预测分类与真实分类进行比较
            errorCount += 1 # 如果分类错误则错误数+1
            # print docList[docIndex]
            # print '预测分类是', predictClass, ', 但实际分类却是', classList[docIndex]
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocabList, p0V, p1V # 返回词汇表, 类别为0的概率, 类别为1的概率

def getTopWords(ny, sf):
    '''
    获取最具表征性的地域词汇
    :param ny:
    :param sf:
    :return:
    '''
    vocabList, pSF, pNY = localWords(ny, sf) # 生成词汇表, newyork地区出现的单词的概率, sfbay地区出现的单词的概率
    topNY = [] # newyork地区高频词
    topSF = [] # sfbay地区高频词
    for i in range(len(pSF)):
        if (pSF[i] > -5): # 条件概率大于-5
            topSF.append((vocabList[i], pSF[i]))
        if (pNY[i] > -5): # 条件概率大于-5
            topNY.append((vocabList[i], pNY[i]))
    # 按照条件概率从大到小进行排序
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print 'SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**'
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print 'NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**'
    for item in sortedNY:
        print item[0]

if __name__=="__main__":
    # testingNB()
    # spamTest()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)