# 评估体系

- 机器学习评估体系说明

## 评估体系说明

# Score模块使用说明
### 运行主函数：
- confusion_matrix_score(file_dir, type='1', save_res_dir='')
  - file_dir: 数据地址；
  - type: 评估的模型类型,'1'为普通多分类评估类型，'2'为机器学习特殊多分类的问题，
      如需要分词标注的命名实体识别；
  - save_res_dir: 结果的保存路径，保存文件必须为csv格式；

### 评估打分需要的数据：
- 模型真实和预测标签文件，根据文件读取真实和预测标签，分别存入列表
- 文件内格式：每行包含两列，第一列为真实标签，第二列为预测标签，两列必须以空格为分隔符；

# Roc模块使用说明
### 运行主函数
- from Roc.roc_curve import get_roc_curve
- get_roc_curve(labels, rates, pos_label=None)
  - labels: 真实测试数据的标签列表；
  - rates: 预测数据标签的概率；
  - pos_label: 如果标签不是0或者1，则pos_label的值为正例的标签 int/str；

### 返回值
- fpr, tpr, thre, roc_auc，Gini
  - fpr: 假正例率,预测为正例但真实情况为反例的，占所有真实情况中反例的比率；
  - tpr: 真正例率,预测为正例且真实情况为正例的，占所有真实情况中正例的比率;
  - thre: 阈值
  - roc_auc: roc曲线所形成的面积，面积越大证明模型越好
  - Gini: **基尼系数**

### Roc曲线的使用意义
1. 能反映模型在选取不同阈值的时候其敏感性（sensitivity, FPR）和其精确性（specificity, TPR）的趋势走向；
2. https://www.jianshu.com/p/2ca96fce7e81

# Ks模块使用说明
### 运行主函数
- from Ks.ks_curve import get_ks_curve
- get_ks_curve(labels, rates, pos_label=None)
  - labels: 真实测试数据的标签列表；
  - rates: 预测数据标签的概率；
  - pos_label: 如果标签不是0或者1，则pos_label的值为正例的标签 int/str；

### 返回值
- fpr, tpr, thre, roc_auc
  - fpr: 假正例率,预测为正例但真实情况为反例的，占所有真实情况中反例的比率；
  - tpr: 真正例率,预测为正例且真实情况为正例的，占所有真实情况中正例的比率;
  - thre: 阈值
  - ks: tpr-fpr的最大值；

### Ks曲线的使用意义
1. 好客户与坏客户的差值(tpr-fpr),值越大说明模型效果越好；
2. https://www.jianshu.com/p/41f434818ffc

# Changelog
- 2020-03-18
  1. 增加Ks曲线模块；
  2. Roc曲线模块中加入基尼系数的计算；
- 2020-03-17
  1. 增加Roc曲线模块；
- 2020-03-11
  1. 新增两种获取混淆矩阵及计算的方法：
         - 机器学习常见多分类问题；
         - 机器学习特殊多分类的问题，如需要分词标注的命名实体识别，只需保留标签不需要
           开头和结尾部分，例：B-TAR需要变为TAR；
- 2020-03-09
  1. 增加Score模块，评估计算机器学习模型的精确度、召回率、F1值；
