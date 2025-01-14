import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages

#读取训练集数据
file_path = r'csv/train/0.xlsx'   # r对路径进行转义，windows需要
raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
raw_data=raw_data.values
raw_data=raw_data[:,1:raw_data.shape[1]]
x0=raw_data
y=np.zeros(raw_data.shape[0])

file_path1='csv/train/1.xlsx'
raw_data1=pd.read_excel(file_path1,header=0)
raw_data1=raw_data1.values
raw_data1=raw_data1[:,1:raw_data1.shape[1]]
x1=raw_data1
y1=np.ones(raw_data1.shape[0])

#x进行竖向拼接，y横向拼接
x=np.vstack((x0,x1))
y_train=np.hstack((y,y1))

#对数据进行归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x)
x0=scaler.fit_transform(x0)
x1=scaler.fit_transform(x1)

#读取验证集数据
file_path = r'csv/validation/0.xlsx'   # r对路径进行转义，windows需要
raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
raw_data=raw_data.values
raw_data=raw_data[:,1:raw_data.shape[1]]
x=raw_data
y=np.zeros(raw_data.shape[0])

file_path1='csv/validation/1.xlsx'
raw_data1=pd.read_excel(file_path1,header=0)
raw_data1=raw_data1.values
raw_data1=raw_data1[:,1:raw_data1.shape[1]]
xt=raw_data1
y1=np.ones(raw_data1.shape[0])

#x进行竖向拼接，y横向拼接
x=np.vstack((x,xt))
y_test=np.hstack((y,y1))

#对数据进行归一化
scaler = MinMaxScaler()
x_test = scaler.fit_transform(x)
# x_test=x_test[:,columns_index]





# #通过T检验从106个特征筛选出44个特征
# from scipy.stats import levene, ttest_ind
# columns_index =[]
# for column_name in range(x1.shape[1]):
#     if levene(x0[:,column_name], x1[:,column_name])[1] > 0.05:
#         if ttest_ind(x0[:,column_name],x1[:,column_name],equal_var=True)[1] < 0.05:
#             columns_index.append(column_name)
#     else:
#         if ttest_ind(x0[:,column_name],x1[:,column_name],equal_var=False)[1] < 0.05:
#             columns_index.append(column_name)
#
# print("筛选后剩下的特征数：{}个".format(len(columns_index)))
# print(columns_index)
# x_train=x_train[:,columns_index]
# x_test=x_test[:,columns_index]
# print(x_train.shape)
# print(x_test.shape)

# 初始化Lasso回归模型，并设置alpha值（正则化强度）
alphas = np.logspace(-3,1,30)# 可以调整这个值，值越大，特征选择越严格

# 训练模型
lasso = LassoCV(alphas = alphas, cv = 10, max_iter = 100000).fit(x_train,y_train)
lasso_best_alpha = lasso.alpha_
print(lasso_best_alpha)
# 获取特征的系数
coef = lasso.coef_
print(coef)
# 找出哪些特征的系数不为零
selected_features = np.where(coef != 0)[0]

print("Selected features:")
print(selected_features)#[ 2  5 19 38 84]

x_train=x_train[:,selected_features]
x_test=x_test[:,selected_features]
print(x_train.shape)
print(x_test.shape)
file_path = r'csv/alldata.csv'   # r对路径进行转义，windows需要
raw_data = pd.read_csv(file_path, header=0)
raw_data=raw_data.drop('name', axis=1, inplace=False).columns.values
print(raw_data[selected_features])
x = raw_data[selected_features]
y = coef[selected_features]

plt.figure(figsize=(12,8))
plt.xlim((-0.65,0.65))
plt.barh(x, y)
plt.show()

# alpha_lasso = 10 ** np.linspace(-3, 1, 100)
# lasso = Lasso()
# coefs_lasso = []
#
# for i in alpha_lasso:
#     lasso.set_params(alpha=i)
#     lasso.fit(x_train, y_train)
#     coefs_lasso.append(lasso.coef_)
#
# plt.figure(figsize=(12, 10))
# ax = plt.gca()
# ax.plot(alpha_lasso, coefs_lasso)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlim((10**-3,10**-1))
# plt.xlabel('alpha')
# plt.ylabel('weights: scaled coefficients')
# plt.title('Lasso regression coefficients Vs. alpha')
# file_path = r'csv/alldata.csv'   # r对路径进行转义，windows需要
# raw_data = pd.read_csv(file_path, header=0)
# print(raw_data.drop('name', axis=1, inplace=False).columns)
# plt.legend(raw_data.drop('name', axis=1, inplace=False).columns)

# plt.savefig("2.svg", dpi=300,bbox_inches = 'tight')

# RandomFrorest
model_rf = RandomForestClassifier(max_depth=5,n_estimators=100,random_state=1).fit(x_train,y_train)
# model_rf=GradientBoostingClassifier(n_estimators=300,max_depth=2,random_state=1,subsample=0.8,learning_rate=0.01).fit(x_train,y_train)
score_rf = model_rf.score(x_test,y_test)
score = model_rf.score(x_train,y_train)
print(score)
print(score_rf)