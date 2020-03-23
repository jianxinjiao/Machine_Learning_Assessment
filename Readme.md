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

### 返回值
- pd.DataFrame，里面包含数据数量，标签个数，混淆矩阵，各个标签及平均的score，recall，f1值；

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

# Kappa系数使用说明
### 运行主函数
from Kappa.kappa_coefficient import get_kappa_coefficient
- get_kappa_coefficient(file_dir, type='1'):
    - file_dir: 数据地址；
    - type: 评估的模型类型,'1'为普通多分类评估类型，'2'为机器学习特殊多分类的问题，
      如需要分词标注的命名实体识别；

### 评估打分需要的数据：
- 模型真实和预测标签文件，根据文件读取真实和预测标签，分别存入列表
- 文件内格式：每行包含两列，第一列为真实标签，第二列为预测标签，两列必须以空格为分隔符；

### 返回值
- 返回kappa值 --> float

### kappa系数使用的意义及计算公式
- kappa系数是一种衡量分类精度的指标；
- kappa = (po - pe)/ (1 - pe)
    - po：表示为总的分类准确度;
    - pe：真实的样本个数乘以预测出来的样本个数；
- https://blog.csdn.net/wang7807564/article/details/80252362

# 回归模型
### 运行主函数
#### 平均绝对误差
- 平均绝对误差能更好地反映预测值误差的实际情况
- from sklearn.metrics import mean_absolute_error
- mean_absolute_error(y_test,y_predict)
  - y_test: 测试数据，type --> np.array/list
  - y_predict: 预测数据，type --> np.array/list
#### 均方误差
- 这也是线性回归中最常用的损失函数，线性回归过程中尽量让该损失函数最小。
  那么模型之间的对比也可以用它来比较。MSE可以评价数据的变化程度，MSE的值
  越小，说明预测模型描述实验数据具有更好的精确度。
- from sklearn.metrics import mean_squared_error
- mean_squared_error(y_test,y_predict)
  - y_test: 测试数据，type --> np.array/list
  - y_predict: 预测数据，type --> np.array/list
#### R-square(决定系数)
- 数学理解： 分母理解为原始数据的离散程度，分子为预测数据和原始数据的误差，二者相除可以消除原始数据离散程度的影响
  其实“决定系数”是通过数据的变化来表征一个拟合的好坏。
  理论上取值范围（-∞，1], 正常取值范围为[0 1] ------实际操作中通常会选择拟合较好的曲线计算R²，因此很少出现-∞
  越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好
  越接近0，表明模型拟合的越差
  经验值：>0.4， 拟合效果好
  缺点：数据集的样本越大，R²越大，因此，不同数据集的模型结果比较会有一定的误差
- from sklearn.metrics import r2_score
- r2_score(y_test,y_predict)
  - y_test: 测试数据，type --> np.array/list
  - y_predict: 预测数据，type --> np.array/list

### 详情
- https://blog.csdn.net/sinat_16388393/article/details/91427631

# Changelog
- 2020-03-23
  1. 完成一些回归模型评估的说明；
- 2020-03-19
  1. 完成kappa系数模块；
- 2020-03-18
  1. 增加Ks曲线模块；
  2. Roc曲线模块中加入基尼系数的计算；
- 2020-03-17
  1. 增加Roc曲线模块；
- 2020-03-11
  1. 新增两种获取混淆矩阵及计算的方法：
     - 机器学习常见多分类问题；
     - 机器学习特殊多分类的问题，如需要分词标注的命名实体识别，
       只需保留标签不需要开头和结尾部分，例：B-TAR需要变为TAR；
- 2020-03-09
  1. 增加Score模块，评估计算机器学习模型的精确度、召回率、F1值；
