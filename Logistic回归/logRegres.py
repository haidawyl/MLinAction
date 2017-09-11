#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

def loadDataSet():
    '''
    读取文件加载数据集
    :return:
    '''
    dataMat = [] # 数据矩阵
    labelMat = [] # 类别标签向量
    fr = open('testSet.txt') # 打开文件
    for line in fr.readlines(): # 遍历所有行数据
        lineArr = line.strip().split() # 截取掉每行的回车字符, 再使用空格字符 ' ' 将行数据分割成一个元素列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 数据矩阵加入1.0以及文件的第0列和第1列, 生成3维数据[x0, x1, x2]
        labelMat.append(int(lineArr[2])) # 类别标签矩阵加入第2列
    return dataMat, labelMat # 返回数据矩阵和类别标签矩阵

def sigmoid(inX):
    '''
    sigmoid函数, sigma(z) = 1 / (1 + np.exp(-z))
    :param inX:
    :return:
    '''
    # 如果inX是一个向量或数组, 则np.exp(-inX)是针对其中的每一个元素进行运算的, 得到的结果仍然是一个向量或数组
    return np.longfloat(1.0 / (1 + np.exp(-inX)))

def gradAscent(dataMatIn, classLabels):
    '''
    批量梯度上升算法
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :return: 回归系数
    '''
    dataMatrix = np.mat(dataMatIn) # 将数组转换为NumPy矩阵
    labelMat = np.mat(classLabels).transpose() # 将一维数组转换为NumPy行向量, 然后再对行向量进行转置变成列向量
    m, n = np.shape(dataMatrix) # 取得数据矩阵的行数和列数
    alpha = 0.001 # 向目标移动的步长
    maxCycles = 500 # 最大迭代次数
    weights = np.ones((n, 1)) # 回归系数, 创建以1填充的n行x1列的NumPy数组
    # 迭代maxCycles后得到回归系数
    for k in range(maxCycles):
        # dataMatrix: m x n
        # weights: n x 1
        # resultMatrix = dataMatrix * weights: m x 1
        # dataMatrix[0][0] * weights[0] + dataMatrix[0][1] * weights[1] + dataMatrix[0][2] * weights[2] + ... + dataMatrix[0][n] * weights[n] = resultMatrix[0][0]
        # dataMatrix[1][0] * weights[0] + dataMatrix[1][1] * weights[1] + dataMatrix[1][2] * weights[2] + ... + dataMatrix[1][n] * weights[n] = resultMatrix[1][0]
        # dataMatrix[2][0] * weights[0] + dataMatrix[2][1] * weights[1] + dataMatrix[2][2] * weights[2] + ... + dataMatrix[2][n] * weights[n] = resultMatrix[2][0]
        # ...
        # dataMatrix[m][0] * weights[0] + dataMatrix[m][1] * weights[1] + dataMatrix[m][2] * weights[2] + ... + dataMatrix[m][n] * weights[n] = resultMatrix[m][0]
        # 对列向量运行sigmoid函数, 返回的结果还是列向量
        # h: m x 1
        h = sigmoid(dataMatrix * weights) # 得到预测类别的值
        # 向量相减等于向量中对应的元素相减, 得到的结果还是向量
        # labelMat: m x 1
        # error: m x 1
        # 梯度反方向=实际值-预测值
        error = (labelMat - h) # 计算真实类别与预测类别的差值
        # 梯度方向=预测值-实际值
        # error = (h - labelMat) # 计算真实类别与预测类别的差值
        # dataMatrix; m x n
        # dataMatrix.transpose(): n x m
        # error: m x 1
        # dataMatrix.transpose() * error: n x 1 本次计算得到的梯度
        # 数字乘以向量等于该数字分别乘以向量中的每一个元素
        # alpha * dataMatrix.transpose() * error: n x 1
        # weights: n x 1
        # 向量相加等于向量中对应的元素相加, 得到的结果还是向量
        # 梯度上升算法(梯度反方向)
        weights = weights + alpha * dataMatrix.transpose() * error # 按照真实类别与预测类别的差值方向调整回归系数
        # 梯度下降算法(梯度方向)
        # weights = weights - alpha * dataMatrix.transpose() * error  # 按照真实类别与预测类别的差值方向调整回归系数
    return weights # 返回回归系数

def plotBestFit(weights):
    '''
    :param weights: 回归系数
    :return:
    '''
    dataMat, labelMat = loadDataSet() # 读取数据集和类别标签
    dataArr = np.array(dataMat) # 将数据集转换成NumPy数组
    trainingClassEst = np.mat(dataArr) * np.mat(weights)

    fig = plt.figure() # 画图
    ax = fig.add_subplot(111)
    ax.scatter(dataArr[np.array(labelMat) == 1.0][:, 1], dataArr[np.array(labelMat) == 1.0][:, 2], s= 30, c='r', marker='s') # 画标签为1的散点图
    ax.scatter(dataArr[np.array(labelMat) == 0.0][:, 1], dataArr[np.array(labelMat) == 0.0][:, 2], s= 30, c='g') # 画标签为0的散点图
    # 创建区间为[-0.3, 0.3), 步长为0.1的等差numpy.ndarray数组
    x1 = np.arange(-3.0, 3.0, 0.1) # 我们要画的横轴数据
    # 此处令sigmoid函数为0. 0是两个分类(即类别l和类别0)的分界处. 因此我们设定0 = w0x0 + w1x1 + w2x2,
    # 在生成数据时, 我们已将x0设置为1.0, 则有x2 = (-weights[0] - weights[1] * x1) / weights[2]
    x2 = (-weights[0] - weights[1] * x1) / weights[2] # 根据公式计算得出纵轴数据
    ax.plot(x1, x2) # 画出分割线
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    plotROC(np.mat(trainingClassEst).T, labelMat, u'训练集ROC曲线')

def stocGradAscent0(dataMatrix, classLabels, numIter=10):
    '''
    随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 类别标签
    :return: 回归系数
    '''
    m, n = np.shape(dataMatrix) # 取得数据集的行数和列数
    alpha = 0.01 # 向目标移动的步长
    weights = np.ones(n) # 回归系数, 创建以1填充的长度为n的NumPy数组

    # 记录回归系数每次改变后的值的数组
    weightsArr = []
    for i in range(m * numIter): # 遍历numIter次数据集
        # dataMatrix[i]; 长度为n的NumPy数组
        # weights; 长度为n的NumPy数组
        # dataMatrix[i] * weights: 进行点乘, 得到长度为n的NumPy数组
        # np.sum(dataMatrix[i] * weights): 对NumPy数组所有元素加和得到一个数字
        # 调用sigmoid函数, 参数是一个数字, 得到的h也是一个数字
        h = sigmoid(np.sum(dataMatrix[i%m] * weights)) # 得到预测类别的值
        # classLabels: 长度为m的列表
        # classLabels[i]: 列表的一个元素, 因此是一个数字
        # 梯度方向=预测值-实际值
        # 梯度反方向=实际值-预测值
        error = classLabels[i%m] - h # 计算真实类别与预测类别的差值
        # dataMatrix[i]; 长度为n的NumPy数组
        # 一个数字乘以一个NumPy数组等于该数字分别乘以数组中的每一个元素
        # error * dataMatrix[i%m]: 本次计算得到的梯度
        # weights: 长度为n的NumPy数组
        # NumPy数组加上NumPy数组等于数组中对应的元素相加, 得到的结果还是NumPy数组
        # 梯度上升算法
        weights = weights + alpha * error * dataMatrix[i%m] # 按照真实类别与预测类别的差值方向调整回归系数

        weightsArr.append(weights.copy())

    # 画出回归系数的变化情况
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(311) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    indexArr = range(len(weightsArr))
    plt.plot(indexArr, np.mat(weightsArr)[:, 0], alpha=0.5)
    plt.ylabel('X0')
    plt.subplot(312)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(indexArr, np.mat(weightsArr)[:, 1], alpha=0.5)
    plt.ylabel('X1')
    plt.subplot(313)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(indexArr, np.mat(weightsArr)[:, 2], alpha=0.5)
    plt.ylabel('X2')
    plt.show()

    return weights # 返回回归系数

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    改进后的随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 类别标签
    :param numIter: 迭代次数
    :return: 回归系数
    '''
    m, n = np.shape(dataMatrix) # 取得数据集的行数和列数
    weights = np.ones(n) # 回归系数, 创建以1填充的长度为n的NumPy数组

    # 记录回归系数每次改变后的值的数组
    weightsArr = []
    for j in range(numIter): # 遍历numIter次
        dataIndex = range(m) # 生成[0, m)的整数列表, 即为数据集索引值的列表
        for i in range(m): # 遍历整个数据集
            # 虽然alpha在每次迭代时都会减小, 但是永远也不会减小到0, 这是因为存在一个常数项.
            # 这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响. 如果要处理的问题
            # 是动态变化的, 那么可以适当加大该常数项, 来确保新的值获得更大的回归系数.
            # 另外值得注意的一点是, 在降低alpha的函数中, alpha每次减少1/(j+i), 其中j是迭代次数,
            # i是样本点的下标. 这样当j<<max(i)时, alpha就不是严格下降的.
            alpha = 4 / (1.0 + j + i) + 0.01 # 每次动态减小alpha值
            # 通过随机选取样本来更新回归系数. 这种方法将减少周期性的波动.
            randIndex = int(random.uniform(0, len(dataIndex))) # 随机生成一个[0, m)范围内的实数, 并转换为整数, 即得到随机选取样本的索引值

            # dataMatrix[randIndex]; 长度为n的NumPy数组
            # weights; 长度为n的NumPy数组
            # dataMatrix[randIndex] * weights: 进行点乘, 得到长度为n的NumPy数组
            # np.sum(dataMatrix[randIndex] * weights): 对NumPy数组所有元素加和得到一个数字
            # 调用sigmoid函数, 参数是一个数字, 得到的h也是一个数字
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights)) # 得到预测类别的值
            # classLabels: 长度为m的列表
            # classLabels[randIndex]: 列表的一个元素, 因此是一个数字
            error = classLabels[randIndex] - h # 计算真实类别与预测类别的差值
            # dataMatrix[randIndex]; 长度为n的NumPy数组
            # 一个数字乘以一个NumPy数组等于该数字分别乘以数组中的每一个元素
            # error * dataMatrix[i%m]: 本次计算得到的梯度
            # weights: 长度为n的NumPy数组
            # NumPy数组加上NumPy数组等于数组中对应的元素相加, 得到的结果还是NumPy数组
            weights = weights + alpha * error * dataMatrix[randIndex] # 按照真实类别与预测类别的差值方向调整回归系数
            del(dataIndex[randIndex]) # 删除dataIndex列表中已经使用过的元素, 下次计算梯度时不再使用该样本, 保证所有的样本数据都能够参与计算梯度

            weightsArr.append(weights.copy())

    # 画出回归系数的变化情况
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(311) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    indexArr = range(len(weightsArr))
    plt.plot(indexArr, np.mat(weightsArr)[:, 0], alpha=0.5)
    plt.ylabel('X0')
    plt.subplot(312)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(indexArr, np.mat(weightsArr)[:, 1], alpha=0.5)
    plt.ylabel('X1')
    plt.subplot(313)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.plot(indexArr, np.mat(weightsArr)[:, 2], alpha=0.5)
    plt.ylabel('X2')
    plt.show()

    return weights # 返回回归系数

def testGradAscent():
    '''
    测试批量梯度上升算法
    :return:
    '''
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights.getA())
    print weights

def testStocGradAscent0():
    '''
    测试随机梯度上升算法
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent0(np.array(dataArr), labelMat, numIter=200)
    plotBestFit(np.mat(weights).T.getA())
    print weights

def testStocGradAscent1():
    '''
    测试随机梯度上升算法
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataArr), labelMat, numIter=30)
    plotBestFit(np.mat(weights).T.getA())
    print weights

def classifyVector(inX, weights):
    '''
    执行分类操作
    :param inX: 特征向量
    :param weights: 回归系数
    :return:
    '''
    # 特征向量点乘回归系数, 再对乘积进行加和运算, 最后使用得到的和值调用Sigmoid算法得到预测值
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5: # 预测值大于0.5, 归入类别1
        return 1.0
    else: # 否则归入类别0
        return 0.0

def colicTest():
    '''
    :return:
    '''
    frTrain = open('horseColicTraining.txt') # 打开训练集文件
    frTest = open('horseColicTest.txt') # 打开测试集文件
    trainingSet = [] # 训练集
    trainingLabels = [] # 训练集类别标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t') # 截取掉每行的回车字符, 再使用tab字符 '\t' 将行数据分割成一个元素列表
        lineArr = []
        for i in range(21): # 前21列是特征数据
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) # 最后一列是类别标签
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels, 500) # 使用训练集计算回归系数向量
    errorCount = 0 # 预测错误数
    numTestVec = 0.0 # 测试样本数量
    # 导入测试集
    for line in frTest.readlines():
        numTestVec += 1.0 # 测试样本数量+1
        currLine = line.strip().split('\t') # 截取掉每行的回车字符, 再使用tab字符 '\t' 将行数据分割成一个元素列表
        lineArr = []
        for i in range(21): # 前21列是特征数据
            lineArr.append(float(currLine[i]))
        # 预测分类, 与实际分类不符, 错误总数+1
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    # 计算错误率
    errorRate = (float(errorCount) / numTestVec)
    print 'the error rate of this test is : %f' % errorRate
    return errorRate

def multiTest():
    '''
    执行10次训练和测试, 计算平均错误率
    :return:
    '''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests))

def drawSigmoid():
    '''
    :return:
    '''
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white', figsize=(6, 5)) # 创建一个新图形, 背景色设置为白色
    plt.subplot(211) # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    x1 = np.arange(-5.0, 6.0, 0.1)
    y1 = sigmoid(x1)
    plt.plot(x1, y1)

    plt.subplot(212)  # subplot(numRows, numCols, plotNum) 将整个绘图区域等分为numRows行* numCols列个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    x2 = np.arange(-60, 60, 1)
    y2 = sigmoid(x2)
    plt.plot(x2, y2)

    plt.show()

def plotROC(predStrengths, classLabels, title):
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
    plt.title(title)
    ax.axis([0, 1, 0, 1])
    plt.show()
    print 'the Area Under the Curve is: ', ySum*xStep

if __name__=='__main__':
    # testGradAscent()
    # testStocGradAscent0()
    testStocGradAscent1()
    # multiTest()
    # drawSigmoid()