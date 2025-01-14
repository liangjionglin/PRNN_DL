# -*- coding: utf-8 -*-
import os
os.environ["KERAS_BACKEND"] = "jax"  # or 'tensorflow' Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import numpy as np
import tensorflow as tf  # for data preprocessing

# import keras_cv
import keras
import cv2
from keras import layers,ops
import keras_cv
from scipy import ndimage
from tqdm import tqdm
from videoswin import *
width=224
height=224
depth= 16
batch_size = 8
epochs = 0
import random

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 3))

    x = ops.transpose(inputs,[0,3,2,1,4])
    model = VideoSwinT(
        num_classes=400,
        include_rescaling=True,
        input_shape=(depth, width, height, 3),
        activation=None
    )
    model.load_weights( 'videoswin_tiny_kinetics400_classifier.weights.h5')

    backbone = keras.Model(model.inputs,model.layers[-2].output,name='3d-swin-transformer')

    x = backbone(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3d-classfier")
    return model

'''
def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 3))
    backbone = keras_cv.models.CSPDarkNetBackbone.from_preset(
        "csp_darknet_tiny_imagenet",
    )
    x = ops.transpose(inputs,[0,3,2,1,4])
    x = ops.reshape(x,[-1,width, height, 3])
    x = backbone(x)
    dims = x.shape[-1]
    x = ops.reshape(x,[-1,depth]+list(x.shape[1:]))
    x = ops.transpose(x,[0,3,2,1,4])
    
    x = layers.Conv3D(filters=dims, kernel_size=2, activation="swish")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=dims, kernel_size=2, activation="swish")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=dims, activation="swish")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model
'''
def resize_volume(img,desired_width,desired_height,desired_depth):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    _depth = current_depth / desired_depth
    _width = current_width / desired_width
    _height = current_height / desired_height
    depth_factor = 1 / _depth
    width_factor = 1 / _width
    height_factor = 1 / _height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor,1), order=1)
    return img
# Build model.
model = get_model(width, height, depth)
model.summary()


def load_data(dirs_path,label):
    x = []
    paths = os.listdir(dirs_path)[:]
    for path in tqdm(paths):
        try:
            t =  os.listdir(dirs_path+path+'/')
            files = []
            for a in t:
                if 'png'  in a and 'T1C'  in a:
                    files.append(a)
        except:
            continue
        image =[]
        for file in random.sample(files,depth) if len(files)>depth else files:
            try:
                img = cv2.imread(dirs_path+path+'/'+file)
            
            except:
                continue
            image.append(img)
        image = np.transpose(image,[1,2,0,3])
        x.append(resize_volume(image,width,height,depth))
    return np.array(x,'float32')

#路径自己写
abnormal_scans = load_data('dataset/0/',0)
normal_scans = load_data('dataset/1/',1)

print(abnormal_scans.shape)
print(normal_scans.shape)
print('For the MRI scans having presence of viral pneumoniaassign 1, for the normal ones assign 0.')
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

print(' Split data in the ratio 70-30 for training and validation.')
x_val = np.concatenate((abnormal_scans[:68], normal_scans[:63]), axis=0)
y_val = np.concatenate((abnormal_labels[:68], normal_labels[:63]), axis=0)
x_train = np.concatenate((abnormal_scans[68:], normal_scans[63:]), axis=0)
y_train = np.concatenate((abnormal_labels[68:], normal_labels[63:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

import random

from scipy import ndimage


def rotate(volume):
    """Rotate the volume by a few degrees"""
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20,25,30,-25,-30,12.5,-12.5,35,-35,40,-40,-45,45]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, reshape=False)
    return volume


def train_preprocessing(volume):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    
    volume = rotate(volume)
    return volume


def vaild_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    #volume = rotate(volume)
    return volume, label
from bert4keras3.snippets import DataGenerator

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        xs,ys =[],[]
        for is_end, (x, y) in self.sample(random):
            ys.append(y)
            xs.append(train_preprocessing(x))
            if len(xs) == self.batch_size or is_end:
                yield np.array(xs), np.array(ys)
                xs,ys =[],[]
datas = []
for i in range(len(x_train)):
    datas.append([x_train[i],y_train[i]])
train_dataset =  data_generator(datas,batch_size=batch_size)   

import matplotlib.pyplot as plt

image = x_train[0]

print("Dimension of the MRI scan is:", image.shape)
plt.imshow(image[:, :, 0], cmap="gray")

def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 MRI slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the MRI scan.
plot_slices(4, -1, width, height, image)







# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.keras", monitor="val_acc", verbose=1, save_best_only=True
)

class CustomSensitivitySpecificityCallback(tf.keras.callbacks.Callback):    
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
            model.save_weights('best_model.weights.h5')
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
    decay_steps = x_train.shape[0]//batch_size*epochs
    initial_learning_rate = 5e-5
    lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps,warmup_steps=x_train.shape[0]//batch_size)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Lamb(learning_rate=lr_decayed_fn,weight_decay=1e-4),
        metrics=["acc"],
    )
    model.fit(
        train_dataset.forfit(),#x_train,y_train,
        validation_data=(x_val, y_val),
        steps_per_epoch=len(train_dataset),
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        verbose=1,
        callbacks=[checkpoint_cb, sensitivity_specificity_cb],
    )

print("\n现在直接使用加载的权重(epochs=0) 或 最后一次训练后的权重(epochs>0)再算一下各个指标")
# 获取验证集的预测结果
model.load_weights('best_model_763_tiny.weights.h5')
y_pred = model.predict(x_val,batch_size=batch_size)
# 将预测概率转换为类别标签（0或1）
best_acc = 0
from tqdm import trange
step_num = int(1e5)
for i in trange(step_num ):
    t=i/step_num 
    y_pred_label = [1 if i >= t else 0 for i in y_pred]
    acc = sum(np.reshape(y_pred_label,y_val.shape)==y_val)/len(y_val)
    if acc>best_acc:
        best_acc = acc
        threshold = t
print('threshold is ',threshold)
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

x = [np.concatenate((abnormal_scans, normal_scans), axis=0)]
y = np.concatenate((abnormal_labels,
                          normal_labels), axis=0)
y_pred = model.predict(x)
result =[]
n = 0
paths = os.listdir( 'dataset/0/')[:]
for i,path in tqdm(enumerate(paths)):
    if y[n]!=1:
        raise(1)
    if i <68:
        label = 'vaild'
    else:
        label = 'train'
    result.append([int(path),y_pred[n][0],1,label])
    n+=1
paths = os.listdir( 'dataset/1/')[:]
for i,path in tqdm(enumerate(paths)):
    if y[n]!=0:
        raise(1)
    if i <63:
        label = 'vaild'
    else:
        label = 'train'
    result.append([int(path),y_pred[n][0],0,label])
    n+=1
    
result = np.array(result)
import pandas as pd
pd.DataFrame(result).to_csv('MRI_result.csv')