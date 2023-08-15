# 所有的的评价指标，都是可以通过混淆矩阵得到的
import torch
import numpy as np
np.set_printoptions(suppress=True)  # 取消科学计数法
import cv2
import torch.nn.functional as F
__all__ = ['Evaluator']

""" 计算混淆矩阵
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""

class Evaluator(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2) 

    def genConfusionMatrix(self, imgPredict, imgLabel):  
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape  # 如果两边相等，可以顺利通过，否则退出
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 调用混淆矩阵函数
        return self.confusionMatrix

    def reset(self):          # 重置混淆矩阵
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

# ------------------------------------------------评价指标---------------------------------------------------------------
    def pixelAccuracy(self):

        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()  # diag取出矩阵的对角线元素
        return acc
    def classPixelAccuracy(self):

        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc 
    def classPixelRecall(self):

        # acc = (TP) / TP + FN
        classRecall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classRecall  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的召回率率
    def classF1(self):
        # return each category pixel F1-score(A more Recall way to call it F1-score)
        # F1-score = 2PR / P + R
        P = self.classPixelAccuracy()
        R = self.classPixelRecall()
        classF1 = 2*P*R / (P+R)
        return classF1  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的F1-score
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  
        return meanAcc



