# 18-决策树(中): CART， 一棵是回归树，另一颗是分类树



CART(Classification And Regression Tree)， 中文名叫做分类回归树



ID3 和 C4.5 可以生成二叉树或多叉树

CART 只支持二叉树，同时，CART 既可以做分类树，又可以作回归树

![image-20190904183910304](/Users/lirawx/Library/Application Support/typora-user-images/image-20190904183910304.png)

、

## CART 分类树工作流程

ID3 信息增益

C4.5 信息增益率

CART 采用基尼系数选择指标



![image-20190904184300487](/Users/lirawx/Library/Application Support/typora-user-images/image-20190904184300487.png)



### 如何使用CART 算法来创建分类树

Python  sklearn 中的 DecisionTreeClassifier 类

参数 criterion 默认 'gini'，采用基尼系数



```python
# CART 算法

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

## 准备数据
iris = load_iris()
# 获取特征集和分类标识
features = iris.data
labels = iris.target

# 随机抽取 33% 的数据作为测试集，其余作为训练集

train_features, test_features,train_labels, test_labels = train_test_split(features, 
                                             labels, 
                                             test_size=0.33, 
                                             random_state=24)
# 创建CART 分类树
clf = DecisionTreeClassifier(criterion='gini')
# 拟合构造 CART 分类树
clf = clf.fit(train_features, train_labels)
# 用 CART 作预测
test_predict = clf.predict(test_features)

# 预测结果比对
score = accuracy_score(test_labels, test_predict)
print("CART 分类树准确率 %.4lf" % score)
```



![image-20190904185538681](/Users/lirawx/Library/Application Support/typora-user-images/image-20190904185538681.png)



## CART 回归树的工作流程

CART 分类树采用 基尼系数作为标准

CART 回归树采用 样本的离散程度



![image-20190904191217438](/Users/lirawx/Library/Application Support/typora-user-images/image-20190904191217438.png)



### 如何使用 CART 回归树作预测



```python

# CART 算法
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
# from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn import utils

##  准备数据
boston = load_boston()

# 探索数据
print(boston.feature_names)

# 获取特征集和房价
features = boston.data
prices = boston.target

# 随机抽取 33% 的数据作为测试集，其余为训练集
train_features, test_features,train_prices, test_prices = train_test_split(features, 
                                             prices, 
                                             test_size=0.33, 
                                             random_state=24)
# 创建 CART 回归树

dtr = DecisionTreeClassifier()

# 拟合构造 CART 回归树
lab_enc = preprocessing.LabelEncoder()
# passing floats to a calssifier 可以被分类
train_prices= lab_enc.fit_transform(train_prices)
dtr.fit(train_features, train_prices)

# 预测测试集中的房价

predict_price = dtr.predict(test_features)

# 测试集的结果评价
print('回归树的二乘偏差均值:', mean_squared_error(test_prices,predict_price))
print('回归树的绝对值偏差均值:', mean_absolute_error(test_prices,predict_price))
```



## CART 决策树的剪枝

CART决策树的剪枝主要采用CCP方法, 它是一种后剪枝的方法,英文全称叫做 cost-complexity  prune, 中文 叫做代价复杂度。

![image-20190904193458397](/Users/lirawx/Library/Application Support/typora-user-images/image-20190904193458397.png)



![image-20190904193517082](/Users/lirawx/Library/Application Support/typora-user-images/image-20190904193517082.png)



