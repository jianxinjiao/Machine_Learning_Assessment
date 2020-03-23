# -*- coding: utf8 -*-
from collections import Counter
import numpy as np

from exceptions import AssessmentValueException


def get_kappa_coefficient(file_dir, type='1'):
    """
    获取Kappa系数，返回float，利用真实标签与预测标签生成混淆矩阵，然后计算Kappa系数；
    Kappa计算公式：
                  kappa = (po - pe)/ (1 - pe)
                  po：表示为总的分类准确度;
                  pe：真实的样本个数乘以预测出来的样本个数；
    :param file_dir: 模型真实和预测标签文件，根据文件读取真实和预测标签，分别存入列表
                     文件内格式：每行包含两列，第一列为真实标签，第二列为预测标签，两列必须以空格为分隔符；
    :param type: 计算的类型，str
                 '1' --> 普通多分类问题，标签预测正确或者预测为其他标签；
                 '2' --> 数据像BIOES标注问题，标签错位现象；
                 '3' --> 待定；
    :return: kappa --> float
    """
    if type not in ['1', '2']:
        raise AssessmentValueException('请根据需要输入正确 type 值')
    confMatrix = create_confusion_matrix(file_dir, type=type) # 获取混淆矩阵
    total_sum = int(confMatrix.sum())  # 数据总量, 防止相乘时内存溢出
    sum_po = 0
    sum_pe = 0
    for i in range(len(confMatrix[0])):
        sum_po += confMatrix[i][i]  # 预测准确的数量（对角线）
        row = confMatrix[i, :].sum()
        col = confMatrix[:, i].sum()
        sum_pe += row * col
    po = sum_po / total_sum
    pe = sum_pe / (total_sum * total_sum)
    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4)


def get_label_data(file_dir):
    """
    :param file_dir: 数据文件路径
    :return: 返回两个标签列表，例如：['true', 'false',...,'true']
    """
    true_labels, predict_labels = list(), list()
    with open(file_dir, 'r', encoding='utf-8') as f:
        data_str = f.read()
    data_list = data_str.split('\n')
    a = [[true_labels.append(x.split(' ')[0]), predict_labels.append(x.split(' ')[1])] for x in data_list if x]  # 去掉空行
    return (true_labels, predict_labels)


def create_confusion_matrix(file_dir, type='1'):
    """创建混淆矩阵"""
    true_labels, predict_labels = get_label_data(file_dir)
    labels = list(set(true_labels))
    labels2idx = dict(zip(labels, range(len(labels))))
    confMatrix = np.zeros([len(labels2idx), len(labels2idx)], dtype=np.int32)
    len_true_labels = len(true_labels)
    if type == '1':
        for i in range(len_true_labels):
            true_idx = labels2idx[true_labels[i]]
            pre_idx = labels2idx[predict_labels[i]]
            confMatrix[true_idx][pre_idx] += 1
        return confMatrix
    elif type == '2':
        true_list = list()  # 存放真实标签前几个相同标签
        for label in true_labels:
            if not true_list or true_list[-1] == label:  # or，只要前面不满足才会往后判断所以不会出错
                true_list.append(label)
                # 处理最后一组标签
                if len(predict_labels) == len(true_list):
                    confMatrix = matrix_func(true_list, predict_labels, labels2idx, confMatrix)  # 详情见matrix_func
            else:
                pre_list = predict_labels[:len(true_list)]  # 截断与真实标签相等的个数，判断个数
                confMatrix = matrix_func(true_list, pre_list, labels2idx, confMatrix)  # 详情见matrix_func
                predict_labels = predict_labels[len(true_list):]  # 预测标签截断
                true_list = [label]  # 存储新标签

        return confMatrix
    else:
        raise ValueError('请根据需要输入正确 type 值')


def matrix_func(true_list, pre_list, labels2idx, confMatrix):
    """
    辅助 create_confusion_matrix 函数 type='2'的情况，
    减少代码冗余度，
    根据一组真实标签与预测出来的标签进行对比，进而生成填充混淆矩阵，
    预测正确率达到3/5就断预测准确，如果达不到，则认为预测标签中出现最多的标签为误判标签；
    :param true_list: 一组真实的标签；
    :param pre_list: 与真实标签照应且等长的预测标签；
    :param labels2idx: 标签字典；
    :param confMatrix: 混淆矩阵
    :return: 更改后的混淆矩阵
    """
    equal_num = pre_list.count(true_list[-1])  # 预测标签中与真实标签相等的个数
    true_idx = labels2idx[true_list[-1]]  # 真实标签所在的位置
    if equal_num / len(true_list) >= 3 / 5:  # 预测标签中，预测正确率达到3/5就断预测准确
        confMatrix[true_idx][true_idx] += 1
    else:
        c = Counter(pre_list)  # 查找预测中最多的标签，为了填补混淆矩阵；
        max_label = c.most_common(1)[0][0]
        pre_idx = labels2idx[max_label]  # 预测标签所在的位置
        confMatrix[true_idx][pre_idx] += 1
    return confMatrix