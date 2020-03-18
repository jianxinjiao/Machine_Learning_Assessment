# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
from sklearn import metrics

from exceptions import AssessmentValueException


def get_ks_curve(labels, rates, pos_label=None):
    """
    获取ks曲线，仅适用于二分类中
    :param labels: 真实测试数据的标签列表
    :param rates: 预测数据标签的概率
    :param pos_label: 如果标签不是0或者1，则pos_label的值为正例的标签 int/str
    :return: fpr, tpr, thre, roc_auc
             fpr: 假正例率,预测为正例但真实情况为反例的，占所有真实情况中反例的比率；
             tpr: 真正例率,预测为正例且真实情况为正例的，占所有真实情况中正例的比率;
             thre: 阈值；
             ks: tpr-fpr的最大值；
    """
    if len(labels) != len(rates):
        raise AssessmentValueException('标签与概率不等长！')
    # 真实例、假实例的计算
    fpr, tpr, thre = metrics.roc_curve(labels, rates, pos_label=pos_label)
    ks = max(tpr-fpr)
    # 画出ks曲线
    plt.figure()
    lw = 2
    plt.figure(figsize=(9, 8))
    plt.plot(fpr, color='darkorange', lw=lw, linestyle='--',label='fpr')
    plt.plot(tpr, color='navy', lw=lw, label='tpr')
    plt.plot(tpr-fpr, color='blue', lw=lw, label='ks')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ks-curve')
    plt.legend(loc="upper left")
    plt.show()

    return (fpr, tpr, thre, ks)
