#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
from Tkinter import *
import regTrees
import matplotlib as mpl
mpl.use('TkAgg') # 将Matplotlib的后端设定为TkAgg
# 将TkAgg和Matplotlib图链接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    '''
    画出数据点和模型拟合线
    :param tolS: 用户输入的最小误差值
    :param tolN: 用户输入的最少样本数
    :return:
    '''
    # pass # 空语句, 是为了保持程序结构的完整性
    reDraw.f.clf() # 清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111) # 添加一个新图
    if chkBtnVar.get(): # 复选框被选中, 构建模型树
        if tolN < 2: # 最少样本数不能少于2
            tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else: # 复选框未选中, 构建回归树
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].flatten().A[0], reDraw.rawDat[:, 1].flatten().A[0], s=5) # 画出数据点
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) # 画出模型拟合线
    reDraw.canvas.show()

def getInputs():
    '''
    取得用户输入值, 并进行校验
    :return:
    '''
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print 'enter Integer for tolN'
        tolNentry.delete(0, END) # 清空输入框
        tolNentry.insert(0, '10') # 恢复默认值
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print 'enter Integer for tolN'
        tolSentry.delete(0, END) # 清空输入框
        tolSentry.insert(0, '1.0') # 恢复默认值
    return tolN, tolS

def drawNewTree():
    '''
    点击ReDraw按钮时调用该函数重新画出数据点和模型拟合线
    :return:
    '''
    # pass
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = Tk() # 创建Tk类型的根部件

Label(root, text='Plot Place Holder').grid(row=0, columnspan=3) # grid()函数设定行和列的位置

Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root) # 单行文本输入框
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar() # 按钮整数值, 读取Checkbutton的状态
chBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar) # 复选按钮
chBtn.grid(row=3, column=0, columnspan=2)
# Button(root, text='Quit', fg='black', command=root.quit).grid(row=1, column=2)
reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
reDraw(1.0, 10)
root.mainloop() # 启动事件循环, 使该窗口在众多事件中可以响应鼠标点击动作.
