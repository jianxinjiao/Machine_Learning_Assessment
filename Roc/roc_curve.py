# -*- coding: utf8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from exceptions import AssessmentValueException


def get_roc_curve(labels, rates, pos_label=None):
    """
    获取roc曲线，仅适用于二分类中
    :param labels: 真实测试数据的标签列表
    :param rates: 预测数据标签的概率
    :param pos_label: 如果标签不是0或者1，则pos_label的值为正例的标签 int/str
    :return: fpr, tpr, thre, roc_auc
             fpr: 假正例率,预测为正例但真实情况为反例的，占所有真实情况中反例的比率；
             tpr: 真正例率,预测为正例且真实情况为正例的，占所有真实情况中正例的比率;
             thre: 阈值
             roc_auc: roc曲线所形成的面积，面积越大证明模型越好
    """
    if len(labels) != len(rates):
        raise AssessmentValueException('标签与概率不等长！')
    # 数据根据概率进行排序
    data = pd.DataFrame(index=range(0, len(labels)), columns=('label', 'rate'))
    data['label'] = labels
    data['rate'] = rates
    data.sort_values('rate', inplace=True, ascending=False)
    # 真实例、假实例的计算
    fpr, tpr, thre = metrics.roc_curve(data['label'], data['rate'], pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    # 画出roc曲线
    plt.figure()
    lw = 2
    plt.figure(figsize=(9, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return (fpr, tpr, thre, roc_auc)
