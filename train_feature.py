# -*- coding: utf-8 -*-
import os
os.environ["KERAS_BACKEND"] = "jax"  # or 'tensorflow' Or "jax" or "torch"!


# import keras_cv
import keras
import cv2
import numpy as np
from keras import layers,ops
from scipy import ndimage
from tqdm import tqdm
from videoswin import *
width=224
height=224
depth= 16
batch_size = 8
epochs = 0
one_hot_num = 12
import random
import pandas as pd
datas = pd.read_csv('feature.csv').values
import numpy as np
def norm(index,datas):
    datas[:,index] = np.where(datas[:,index]>0,np.log10([float(t)+1.0 for t in datas[:,index]]),0)/10+np.where(datas[:,index]>0,1,0)
    return datas
datas = norm(-1,datas)

#datas = np.concatenate([datas,datas[:,-1:],datas[:,-1:]],axis=-1)

datas_dict = {}
continue_varaibles = []
for t in datas:

    continue_varaibles.append(t[one_hot_num +3:])
continue_varaibles= np.array(continue_varaibles)#(np.array(continue_varaibles)-np.min(continue_varaibles,-1,keepdims=1))/(np.max(continue_varaibles,-1,keepdims=1)-np.min(continue_varaibles,-1,keepdims=1))

for i,t in enumerate(datas):
    key = int(t[0])
    datas_dict[key]={
        'one_hot' : t[3:one_hot_num+3],
        'continue_varaible':continue_varaibles[i]
    }
def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    

    dims = 32
    onehot_inputs  = keras.Input([one_hot_num])
    continue_inputs  = keras.Input([continue_varaibles.shape[-1]])
    onehot_emb = []
    for i in range(one_hot_num):
        onehot_emb.append(keras.layers.Embedding(5,dims)(onehot_inputs[:,i]))
    onehot_emb = keras.layers.Concatenate(-1)(onehot_emb)
    continue_emb = keras.layers.Dense(continue_varaibles.shape[-1]*dims)(continue_inputs)
    z = keras.layers.Concatenate(-1)([onehot_emb,continue_emb])
    z = keras.layers.Dense(units=z.shape[-1])(z)
    z = keras.layers.Dense(units=z.shape[-1], activation="swish")(z)
    z += keras.layers.Dense(units=z.shape[-1], activation="swish")(z)
    z += keras.layers.Dense(units=z.shape[-1], activation="swish")(z)
    # Define the model.
    
    outputs = keras.layers.Dense(units=1, activation="sigmoid")(z)
    model = keras.Model([onehot_inputs,continue_inputs], outputs, name="3d-classfier")
    return model

# Build model.
model = get_model(width, height, depth)
model.summary()


def load_data(dirs_path,label):
    x = []
    paths = os.listdir(dirs_path)[:]
    z1 = []
    z2 = []
    for path in tqdm(paths):
        if int(path) not in datas_dict.keys():
            continue
        z1.append(datas_dict[int(path)]['one_hot'])
        z2.append(datas_dict[int(path)]['continue_varaible'])
        
    return np.array(z1,'float32'),np.array(z2,'float32')

#路径自己写
abnormal_z1,abnormal_z2 = load_data('dataset/0/',0)
normal_z1,normal_z2 = load_data('dataset/1/',1)

abnormal_labels = np.array([1 for _ in range(len(abnormal_z1))])
normal_labels = np.array([0 for _ in range(len(normal_z1))])

print(' Split data in the ratio 70-30 for training and validation.')
def concat(x1,x2,start,end):
    return 
val_num = 65
x_val = [
    np.concatenate([abnormal_z1[:68],normal_z1[:63]], axis=0),
    np.concatenate([abnormal_z2[:68],normal_z2[:63]], axis=0)]
y_val = np.concatenate((abnormal_labels[:68], normal_labels[:63]), axis=0)
x_train = [
         np.concatenate([abnormal_z1[68:],normal_z1[63:]], axis=0),
    np.concatenate([abnormal_z2[68:],normal_z2[63:]], axis=0)]
y_train = np.concatenate((abnormal_labels[68:],
                          normal_labels[63:]), axis=0)
x_max = np.max(x_train[1],keepdims=1,axis=0)
x_min = np.min(x_train[1],keepdims=1,axis=0)
print(x_max)
print(x_min)
x_val[1] =  np.maximum(np.minimum(x_val[1],x_max),x_min)
x_train[1] = (x_train[1]-x_min)/(x_max-x_min)
x_val[1] = (x_val[1]-x_min)/(x_max-x_min)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train[0].shape[0], x_val[0].shape[0])
)

class CustomSensitivitySpecificityCallback(keras.callbacks.Callback):    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.best_acc = 0
    def on_epoch_end(self, epoch, logs=None):

        # 获取验证集的预测结果
        y_pred = self.model.predict(x_val,batch_size=batch_size)

        # 将预测概率转换为类别标签（0或1）
        y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_val, y_pred_classes)

        # 从混淆矩阵中提取TP, TN, FP, FN
        TN, FP, FN, TP = cm.ravel()

        # 计算灵敏度和特异性
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        # 打印灵敏度和特异性
        # print(f"Epoch {epoch+1}:")
        print(f"灵敏度（Sensitivity）: {sensitivity}")
        print(f"特异性（Specificity）: {specificity}")

        from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

        # 计算准确率
        accuracy = accuracy_score(y_val, y_pred_classes)
        print(f"准确率（Accuracy）: {accuracy}")
        if accuracy>self.best_acc:
            self.best_acc = accuracy
            model.save_weights('best_feature_model.weights.h5')
        print(f"最优准确率（Accuracy）: {self.best_acc}")

        # 计算精确度
        precision = precision_score(y_val, y_pred_classes)
        print(f"精确度（Precision）: {precision}")

        # 计算 F1 分数
        f1 = f1_score(y_val, y_pred_classes)
        print(f"F1 分数（F1 Score）: {f1}")

        # 计算 ROC AUC 值
        roc_auc = roc_auc_score(y_val, y_pred)
        print(f"ROC AUC 值: {roc_auc}")

sensitivity_specificity_cb = CustomSensitivitySpecificityCallback()

# Train the model, doing validation at the end of each epoch

if epochs:
    #initial_learning_rate = 0.00000120
    decay_steps = len(x_train[0])//batch_size*epochs
    initial_learning_rate = 1e-3
    lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Lamb(learning_rate=lr_decayed_fn,weight_decay=1e-2),
        metrics=["acc"],
    )
    model.fit(
        x_train,y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        verbose=1,
        callbacks=[sensitivity_specificity_cb],
    )
print("\n现在直接使用加载的权重(epochs=0) 或 最后一次训练后的权重(epochs>0)再算一下各个指标")
# 获取验证集的预测结果
import matplotlib.pyplot as plt
model.load_weights('best_feature_model.weights.h5')
y_pred = model.predict(x_val,batch_size=batch_size)
# 将预测概率转换为类别标签（0或1）
best_acc = 0
from tqdm import trange
step_num = int(1e5)
threshold = 0.5
y_pred_classes = [1 if i >= threshold else 0 for i in y_pred]
# 计算混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred_classes)
# 从混淆矩阵中提取TP, TN, FP, FN
TN, FP, FN, TP = cm.ravel()
# 计算灵敏度和特异性
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(f"灵敏度（Sensitivity）: {sensitivity}")
print(f"特异性（Specificity）: {specificity}")

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
# 计算准确率

accuracy = accuracy_score(y_val, y_pred_classes)
print(f"准确率（Accuracy）: {accuracy}")
# 计算精确度
precision = precision_score(y_val, y_pred_classes)
print(f"精确度（Precision）: {precision}")
# 计算 F1 分数
f1 = f1_score(y_val, y_pred_classes)
print(f"F1 分数（F1 Score）: {f1}")
# 计算 ROC AUC 值
roc_auc = roc_auc_score(y_val, y_pred)
print(f"ROC AUC 值: {roc_auc}")

# 绘制ROC曲线
from sklearn.metrics import roc_curve, auc
plt.figure()
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# 找到最佳阈值
TPR_minus_FPR = tpr - fpr
max_difference = max(TPR_minus_FPR)
maxindex = TPR_minus_FPR.tolist().index(max_difference)
threshold = thresholds[maxindex]

# 在图上标注最佳阈值
plt.scatter([fpr[maxindex]], [tpr[maxindex]], color='red', marker='o', s=100)
plt.text(fpr[maxindex], tpr[maxindex], f'{threshold:.9f}', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='red')

plt.show()

x = [
    np.concatenate([abnormal_z1,normal_z1], axis=0),
    np.concatenate([abnormal_z2,normal_z2], axis=0)]
y = np.concatenate((abnormal_labels,
                          normal_labels), axis=0)
x[1] =  np.maximum(np.minimum(x[1],x_max),x_min)
x[1] = (x[1]-x_min)/(x_max-x_min)
y_pred = model.predict(x)


result =[]
n = 0
paths = os.listdir( 'dataset/0/')[:]
for i,path in tqdm(enumerate(paths)):
    if y[n]!=1:
        raise(1)
    if i <68:
        label = 'valid'
    else:
        label = 'train'
    result.append([int(path),y_pred[n][0],1,label])
    n+=1
paths = os.listdir( 'dataset/1/')[:]
for i,path in tqdm(enumerate(paths)):
    if y[n]!=0:
        raise(1)
    if i <63:
        label = 'valid'
    else:
        label = 'train'
    result.append([int(path),y_pred[n][0],0,label])
    n+=1
    
result = np.array(result)
import pandas as pd
pd.DataFrame(result).to_csv('feature_result.csv')