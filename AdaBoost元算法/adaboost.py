#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def loadSimpData():
    '''
    :return:
    '''
    dataMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def draw():
    '''
    :return:
    '''
    dataMat, classLabels = loadSimpData()
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure() # 画图
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[np.array(classLabels)[:] == 1.0][:, 0].flatten().A, dataMat[np.array(classLabels)[:] == 1.0][:, 1].flatten().A, s= 30, c='r', marker='o') # 画散点图
    ax.scatter(dataMat[np.array(classLabels)[:] == -1.0][:, 0].flatten().A, dataMat[np.array(classLabels)[:] == -1.0][:, 1].flatten().A, s= 30, c='r', marker='s') # 画散点图
    plt.title(u'单层决策树测试数据')
    plt.show()

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据集进行分类, 分类规则为:
    如果threshIneq等于lt, 则数据集中索引为dimen的特征其特征值小于或者等于阈值threshVal时类别被置为-1类, 否则被置为默认值+1类;
    如果threshIneq等于gt, 则数据集中索引为dimen的特征其特征值大于阈值threshVal时类别被置为-1类, 否则被置为默认值+1类
    :param dataMatrix: 数据矩阵
    :param dimen: 特征索引
    :param threshVal: 阈值
    :param threshIneq: 阈值比较运算符, 只有两个选项'lt'和'gt', 'lt'表示小于或者等于, 'gt'表示大于
    :return: 标识数据集类别的数组
    '''
    m, n = np.shape(dataMatrix) # 数据集的行数和列数
    retArray = np.ones((m, 1)) # 标识数据集类别的数组, 默认值全为+1类
    # 所有在阈值一边的数据被分到类别-1中, 而在另外一边的数据被分到类别+1中
    if threshIneq == 'lt':
        # 数据集中索引为dimen的特征其特征值小于或者等于阈值threshVal时类别被置为-1类, 否则被置为默认值+1类
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 数据集中索引为dimen的特征其特征值大于阈值threshVal时类别被置为-1类, 否则被置为默认值+1类
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray # 返回标识数据集类别的数组

def buildStump(dataArr, classLabels, D):
    '''
    在给定权重向量D的情况下, 遍历数据集找到最佳的单层决策树即预测错误率最小的单层决策树.
    这里的"最佳"是基于数据的权重向量D来定义的.
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param D: 权重向量
    :return: 得到最小错误率的单层决策树的相关信息, 最小的错误率和对应单层决策树所预测的分类结果(+1类, -1类)
    '''
    dataMatrix = np.mat(dataArr) # 数据矩阵
    labelMat = np.mat(classLabels).T # 类别标签列向量
    m, n = np.shape(dataMatrix) # 数据矩阵的行数和列数
    numSteps = 10.0 # 将每一个特征的取值区间(最大值-最小值)划分为等长区间的数量
    bestStump = {} # 存储给定权重向量D之后计算所得到的最佳单层决策树的相关信息
    bestClasEst = np.mat(np.zeros((m, 1))) # 类别预测结果矩阵, 默认值全部为0
    minError = np.inf # 最小错误率, 初始值设为正无穷
    for i in range(n): # 遍历所有特征
        rangeMin = dataMatrix[:, i].min() # 取得第i个特征的最小值
        rangeMax = dataMatrix[:, i].max() # 取得第i个特征的最大值
        stepSize = (rangeMax - rangeMin) / numSteps # 得到步长
        # range(-1, int(numSteps)+1): [-1, 10+1)之间的整数, 即-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        # 遍历每一个阈值(当前特征最小值+第j个步长*步长), 在每一个阈值上构建两棵单层决策树,
        # 第一棵单层决策树将特征值中小于或者等于当前阈值的数据标记为-1类, 其它数据标记为+1类
        # 预测值和实际值进行比较, 计算得到预测错误率, 在该错误率的基础上计算加权错误率
        # 第二棵单层决策树将特征值中大于当前阈值的数据标记为-1类, 其它数据标记为+1类
        # 预测值和实际值进行比较, 计算得到预测错误率, 在该错误率的基础上计算加权错误率
        # 加权错误率=预测分类错误的数据的权重之和
        # 在当前特征中找出加权错误率最小的那棵单层决策树并返回
        for j in range(-1, int(numSteps)+1):
            # 切换阈值比较运算符'lt'和'gt'对数据进行分类
            # 'lt'表示小于或者等于, 'gt'表示大于
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize) # 计算阈值
                # 根据阈值预测分类
                # 如果inequal等于lt, 则数据集中索引为i的特征其特征值小于或者等于阈值threshVal时类别被置为-1类, 否则被置为默认值+1类;
                # 如果inequal等于gt, 则数据集中索引为i的特征其特征值大于阈值threshVal时类别被置为-1类, 否则被置为默认值+1类
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1))) # m行x1列的错误向量, 默认值全为1
                # 预测值和实际值相等时, 对应的错误向量索引位置置为0, 此时错误向量中值为1对应的数据即为错分类的样本
                errArr[predictedVals == labelMat] = 0
                # 错误向量errArr中值为1的元素索引对应的数据集上的数据为被错分的样本, 值为0的元素索引对应的数据集上的数据为被正确分类的样本
                # 权重向量D的转置乘以错误向量errArr得到错分样本的权重之和即为加权错误率. 因为权重向量D的所有元素之和永远等于1.0,
                # 所以加权错误率小于或者等于1.0
                weightedError = D.T * errArr
                # print 'split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError)
                # 查找加权错误率最小的单层决策树并保存到bestStump
                if (weightedError < minError):
                    minError = weightedError # 更新最小错误率
                    bestClasEst = predictedVals.copy() # 更新得到最小错误率的单层决策树所预测的分类结果(+1类, -1类)
                    # 保存得到最小错误率的单层决策树的相关信息
                    bestStump['dim'] = i # 特征索引
                    bestStump['thresh'] = threshVal # 阈值
                    bestStump['ineq'] = inequal # 阈值比较运算符
    return bestStump, minError, bestClasEst # 返回得到最小错误率的单层决策树的相关信息, 最小的错误率和对应单层决策树所预测的分类结果(+1类, -1类)

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    训练数据集中的每一个样本, 并赋予其一个权重, 这些权重构成了向量D. 一开始, 这些权重都初始化成相等值,
    并且权重之和等于1.0. 首先在训练数据集上训练出一个弱分类器并计算该分类器的错误率, 然后在同一数据集上
    再次训练弱分类器. 在分类器的第二次训练当中, 将会重新调整每个样本的权重, 其中第一次分对的样本其权重
    将会降低, 而第一次分错的样本其权重将会升高. 为了从所有弱分类器中得到最终的分类结果, AdaBoost为每个
    分类器都分配了一个权重值alpha, 这些alpha值是基于每个弱分类器的错误率计算得到的. 其中, 错误率error的
    定义为:
    error = 未正确分类的样本数目 / 所有样本的数目
    而alpha的计算公式如下:
    alpha = 0.5 * np.log((1.0 - error) / error)
    计算出alpha值之后, 对权重向量D进行更新, 以使得那些正确分类的样本其权重降低而错分样本的权重升高.
    D的计算方法如下:
    如果某个样本被正确分类, 那么该样本的权重更改为:
    Di(t+1) = Di(t) * np.exp(-alpha) / sum(D)
    而如果某个样本被错分, 那个该样本的权重更改为:
    Di(t+1) = Di(t) * np.exp(alpha) / sum(D)
    在计算出D之后, AdaBoost又开始进入下一轮迭代. AdaBoost算法会不断地重复训练和调整权重的过程, 直到
    训练错误率为0或者迭代次数达到用户的指定值为止.

    初始权重相同的情况下, 找到一个使得错误率最低的分类器, 应用此分类器更新权重向量, 即正确分类的样本
    其权重将会降低, 而错误分类的样本其权重将会升高. 样本的权重越高, 在其被分错的情况下, 得到的错误率
    也就越高. 为了降低错误率则必须找到能够将权重高的样本正确分类的分类器, 找到这样的分类器之后, 将会
    使得权重高的样本能够正确分类. 但是可能导致其它样本被分错进而升高其权重, 这样有可能会导致下次找到
    的分类器使得错误率不是降低而是升高. 在整个迭代过程中, 错误率不一定是严格下降的, 而是有可能震荡下
    降的. AdaBoost的过程就是每一次迭代找到一个最佳的分类器使得样本的总体错误率降低,直到错误率降为0或
    者到达用户指定的迭代次数. AdaBoost算法的一个缺点是有可能会导致过拟合.
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数(即弱分类器个数)
    :return:
    '''
    weakClassArr = [] # 存储每一次迭代得到的预测错误率最小的单层决策树的数组
    m = np.shape(dataArr)[0] # 数据集数量
    # 创建初始值为1/m的m行x1列的权重向量, 该权重向量包含了每个数据点的权重.
    # 在迭代过程中, AdaBoost算法会增加错分样本的权重, 同时降低正确分类样本的权重
    # D是一个概率分布向量, 因此其所有元素之和为1.0.为了满足此要求, 一开始所有元素
    # 都会被初始化为1/m.
    D = np.mat(np.ones((m, 1)) / m)
    # 记录每个数据点的累计类别估计值的m行x1列的列向量, 初始值全部为0
    aggClassEst = np.mat(np.zeros((m, 1)))
    errorRateArray = [] # 保存每次迭代过程中的错误率, 用于画出错误率的变化趋势图
    for i in range(numIt): # 遍历参数给定的迭代次数
        # 遍历数据集找到基于权重向量D的最佳单层决策树即预测错误率最小的单层决策树
        # 函数返回最小错误率的单层决策树, 最小错误率和对应单层决策树所预测的分类结果(+1类, -1类)
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print 'D; ', D.T # 输出权重向量D
        # alpha值为本次单层决策树输出结果的权重, 其计算公式为:
        # alpha = 0.5 * np.log((1.0 - error) / error)
        # 在下面的公式中, max(error, 1e-16)表示确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha # alpha加入到本次得到的单层决策树字典中
        weakClassArr.append(bestStump) # 本次得到的单层决策树加入到weakClassArr字典中
        # print 'classEst: ', classEst.T
        # 根据alpha值更新权重向量D, 以使得那些正确分类的样本其权重降低而错分样本的权重升高
        # D的计算方法如下:
        # 如果某个样本被正确分类, 那么该样本的权重更改为:
        # Di(t + 1) = Di(t) * np.exp(-alpha) / sum(D)
        # 而如果某个样本被错分, 那么该样本的权重更改为:
        # Di(t + 1) = Di(t) * np.exp(alpha) / sum(D)
        # np.multiply(A, B): 向量A和向量B对应的元素相乘(即点乘)
        # 预测类别与实际类别相同即正确分类得到-alpha, 否则得到alpha,
        # 带入下面的公式中即得到正确分类则权重降低, 错误分类则权重升高
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 更新累计类别估计值
        aggClassEst += alpha * classEst
        # print 'aggClassEst: ', aggClassEst.T
        # np.sign(x): -1 if x < 0, 0 if x==0, 1 if x > 0
        # np.sign(aggClassEst) != np.mat(classLabels).T: 两个向量对应元素的值不相等则为True即1, 相等则为False即0
        # 上述结果再与全为1的向量点乘, 得到错分的样本, 即值为1的元素对应的样本数据
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        # aggErrors的所有元素相加得到值为1的元素个数, 再除以总样本数得到错误率
        errorRate = aggErrors.sum() / m
        print 'total error: ', errorRate
        errorRateArray.append(errorRate) # 保存本次迭代的错误率
        if errorRate == 0.0: # 如果错误率为0, 则退出循环
            break

    # 画出错误率变化趋势图, 如果数据量比较大是, 从图中我们可以看到错误率不一定是严格下降的,
    # 而有可能是震荡下降的. 当到达某次迭代以后, 错误率不再下降反而上升了, 这就是过拟合现象.
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    numItArray = range(len(errorRateArray))
    plt.plot(numItArray, errorRateArray)
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'错误率')
    plt.title(u'错误率变化趋势图')
    plt.show()

    return weakClassArr, aggClassEst # 返回最佳单层决策树数组和累计类别估计值

def adaClassify(dataToClass, classifierArr):
    '''
    利用训练出来的多个弱分类器进行分类
    :param dataToClass: 待分类的数据
    :param classifierArr: 训练出来的多个弱分类器数组
    :return: 返回分类结果
    '''
    dataMatrix = np.mat(dataToClass) # 将待分类的数据转换为NumPy矩阵
    m = np.shape(dataMatrix)[0] # 得到数据的个数
    # 记录每个数据点的累计类别估计值的m行x1列的列向量, 初始值全部为0
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 遍历所有弱分类器
    for i in range(len(classifierArr)):
        # 基于stumpClassify()对每个分类器得到数据的一个分类结果
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        # 使用分类结果乘上该分类器的alpha权重然后再累加到aggClassEst
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print aggClassEst
    return np.sign(aggClassEst) # 返回aggClassEst的符号结果, 即-1 if x < 0, 0 if x==0, 1 if x > 0

def loadDataSet(fileName):
    '''
    :param fileName:
    :return:
    '''
    numFeat = len(open(fileName).readline().strip().split('\t')) # 特征数目
    dataMat = [] # 数据矩阵
    labelMat = [] # 类别标签矩阵
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        # 每行最后一列为类别标签, 其它列为特征
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def colicTest():
    trainingDataArr, trainingLabelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(trainingDataArr, trainingLabelArr, 50)
    plotROC(aggClassEst.T, trainingLabelArr)
    testDataArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = adaClassify(testDataArr, classifierArray)
    m = np.shape(testDataArr)[0]
    errorArray = np.mat(np.ones((m, 1)))
    errorCount = errorArray[prediction != np.mat(testLabelArr).T].sum()
    print errorCount
    print 'the error rate of this test is : %f' % (float(errorCount) / m)

def plotROC(predStrengths, classLabels):
    '''
    <0.0, 0.0>: 将所有样例判为反例, 则TP=FP=0
    <1.0, 1.0>: 将所有样例判为正例, 则FN=TN=0
    x轴表示假阳率(FP/(FP+TN)), 在<0.0, 0.0>点假阳率等于0, 在<1.0, 1.0>点假阳率等于1
    y轴表示真阳率(TP/(TP+FN)), 在<0.0, 0.0>点真阳率等于0, 在<1.0, 1.0>点真阳率等于1
    :param predStrengths: 行向量, 表示分类结果的预测强度,
    如果值为负数则值越小被判为反例的预测强度越高, 反之值为正数则值越大被判为正例的预测强度越高
    :param classLabels: 类别标签
    :return:
    '''
    cur = (1.0, 1.0) # 绘制光标的位置, 起始点为右上角<1.0, 1.0>的位置
    ySum = 0.0 # 计算AUC(Area Under the Curve, ROC曲线下面的面积)的值
    numPosClas = sum(np.array(classLabels) == 1.0) # 真实正例的数目
    yStep = 1 / float(numPosClas) # y轴的步长
    # len(classLabels) - numPosClas: 真实反例的数目
    xStep = 1 / float(len(classLabels) - numPosClas) # x轴的步长
    # 分类器的预测强度从小到大排序的索引列表
    # 得到的是样本预测强度从小到大(从负数到正数)的样例的索引值列表
    # 第一个索引指向的样例被判为反例的强度最高
    # 最后一个索引指向的样例被判为正例的强度最高
    sortedIndicies = predStrengths.argsort()
    # print predStrengths[0, sortedIndicies]
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历分类器的预测强度从小到大排序的索引列表
    # 先从排名最低的样例开始, 所有排名更低的样例都被判为反例, 而所有排名更高的样例都被判为正例.
    # 第一个值对应点为<1.0, 1.0>, 而最后一个值对应点为<0.0, 0.0>.
    # 然后, 将其移到排名次低的样例中去, 如果该样例属于正例, 那么对真阳率进行修改;
    # 如果该样例属于反例, 那么对假阳率进行修改.

    # 初始时预测强度最小, 那么所有的样本都被判为正例, 即对应图中右上角的位置.
    # 向后迭代的过程中, 预测强度依次增大, 则排名低(列表前面)的样本被判为反例, 排名高(列表后面)的样本被判为正例.
    # 如果当前样本为真实正例, 在将其预测为反例时, 则为伪反例FN, 根据真阳率=TP/(TP+FN), 因此真阳率下降, 沿y轴下移
    # 如果当前样本为真实反例, 在将其预测为反例时, 则为真反例TN, 根据假阳率=FP/(FP+TN), 因此假阳率下降, 沿x轴左移
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0: # 标签为1.0的类即正例, 则要沿着y轴的方向下降一个步长, 也就是要不断降低真阳率
            delX = 0
            delY = yStep
        else: # 标签为其它(0或者-1)的类即反例, 则要沿着x轴的方向倒退一个步长, 也就是要不断降低假阳率
            delX = xStep
            delY = 0
            # 为了计算AUC, 需要对多个小矩形的面积进行累加. 这些小矩形的宽度是xStep, 因此可以先对所有矩形
            # 的高度进行累加, 最后再乘以xStep得到其总面积. 所有高度的和(ySum)随着x轴的每次移动而渐次增加
            ySum += cur[1]
        # 一旦决定了是在x轴还是y轴方向上进行移动, 就可以在当前点和新点之间画出一条线段
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        # 更新当前点cur
        cur = (cur[0]-delX, cur[1]-delY)
    # 画出左下角到右上角之间的虚线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel(u'假阳率(False Positive Rate)')
    plt.ylabel(u'真阳率(True Positive Rate)')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print 'the Area Under the Curve is: ', ySum*xStep

if __name__=='__main__':
    # draw()

    # dataMat, classLabels = loadSimpData()
    # classifierArray, aggClassEst = adaBoostTrainDS(dataMat, classLabels, 10)
    # print classifierArray
    # print adaClassify([0, 0], classifierArray)
    # print adaClassify([[5, 5], [1, 1]], classifierArray)

    colicTest()