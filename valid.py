# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:40:51 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
import os
os.environ["KERAS_BACKEND"] = "torch"  # or 'tensorflow' Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import numpy as np

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
epochs = 50
one_hot_num = 12
import random
import pandas as pd
datas = pd.read_csv('feature.csv').values
import numpy as np
def norm(index,datas):
    datas[:,index] = np.where(datas[:,index]>0,np.log10([float(t)+1.0 for t in datas[:,index]]),0)/10+np.where(datas[:,index]>0,1,0)
    return datas
datas = norm(-1,datas)
datas_dict = {}
continue_varaibles = []
for t in datas:

    continue_varaibles.append(t[one_hot_num +3:])
continue_varaibles= np.array(continue_varaibles)
for i,t in enumerate(datas):
    key = int(t[0])
    datas_dict[key]={
        'one_hot' : t[3:one_hot_num+3],
        'continue_varaible':continue_varaibles[i]
    }
def get_feature_model():
    onehot_inputs  = keras.Input([one_hot_num])
    continue_inputs  = keras.Input([continue_varaibles.shape[-1]])
    onehot_emb = []
    for i in range(one_hot_num):
        onehot_emb.append(keras.layers.Embedding(5,32)(onehot_inputs[:,i]))
    onehot_emb = keras.layers.Concatenate(-1)(onehot_emb)
    continue_emb = keras.layers.Dense(continue_varaibles.shape[-1]*32)(continue_inputs)
    z = keras.layers.Concatenate(-1)([onehot_emb,continue_emb])
    z = keras.layers.Dense(units=z.shape[-1])(z)
    z = keras.layers.Dense(units=z.shape[-1], activation="swish")(z)
    z += keras.layers.Dense(units=z.shape[-1], activation="swish")(z)
    z += keras.layers.Dense(units=z.shape[-1], activation="swish")(z)
    # Define the model.
    
    outputs = keras.layers.Dense(units=1, activation="sigmoid")(z)
    model = keras.Model([onehot_inputs,continue_inputs], outputs, name="3d-classfier")
    model.load_weights('best_feature_model.weights.h5')
    return model
def get_MRI_model(width=128, height=128, depth=64):
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
    model.load_weights('best_model_763_tiny.weights.h5')
    return model
cmodel = get_MRI_model(width, height, depth)
fmodel = get_feature_model()


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

def load_data(dirs_path,label):
    x = []
    paths = os.listdir(dirs_path)[:]
    z1 = []
    z2 = []
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
        z1.append(datas_dict[int(path)]['one_hot'])
        z2.append(datas_dict[int(path)]['continue_varaible'])
        
    return np.array(x,'float32'),np.array(z1,'float32'),np.array(z2,'float32')

#路径自己写
abnormal_scans,abnormal_z1,abnormal_z2 = load_data('dataset/0/',0)
normal_scans,normal_z1,normal_z2 = load_data('dataset/1/',1)

print(abnormal_scans.shape)
print(normal_scans.shape)
print('For the MRI scans having presence of viral pneumoniaassign 1, for the normal ones assign 0.')
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

print(' Split data in the ratio 70-30 for training and validation.')
x_val = [
    np.concatenate((abnormal_scans[:68], normal_scans[:63]), axis=0),
    np.concatenate([abnormal_z1[:68],normal_z1[:63]], axis=0),
    np.concatenate([abnormal_z2[:68],normal_z2[:63]], axis=0)]
y_val = np.concatenate((abnormal_labels[:68], normal_labels[:63]), axis=0)
x_train = [
    np.concatenate((abnormal_scans[68:], normal_scans[63:]), axis=0),
    np.concatenate([abnormal_z1[68:],normal_z1[63:]], axis=0),
    np.concatenate([abnormal_z2[68:],normal_z2[63:]], axis=0)]
y_train = np.concatenate((abnormal_labels[68:],
                          normal_labels[63:]), axis=0)
x_max = np.max(x_train[-1],keepdims=1,axis=0)
x_min = np.min(x_train[-1],keepdims=1,axis=0)
print(x_max)
print(x_min)
x_val[-1] =  np.maximum(np.minimum(x_val[-1],x_max),x_min)
x_train[-1] = (x_train[-1]-x_min)/(x_max-x_min)
x_val[-1] = (x_val[-1]-x_min)/(x_max-x_min)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train[0].shape[0], x_val[0].shape[0])
)
x = x_val
y = y_val
logits_1 = cmodel.predict(x[0],batch_size=4)
logits_2 = fmodel.predict(x[1:])
best_acc = 0

from tqdm import trange
step_num = 1000
for i in trange(step_num):
    l=i/step_num
    y_pred = l*logits_1+(1-l)*logits_2
    for j in range(step_num):
        t = j/step_num
        y_pred_label = [1 if i >= t else 0 for i in y_pred]
        acc = sum(np.reshape(y_pred_label,y_val.shape)==y_val)/len(y_val)
        if acc>=best_acc:
            best_acc = acc
            threshold = t
            lamb = l
print( best_acc,threshold,lamb)
x = [
    np.concatenate((abnormal_scans, normal_scans), axis=0),
    np.concatenate([abnormal_z1,normal_z1], axis=0),
    np.concatenate([abnormal_z2,normal_z2], axis=0)]
x[-1] =  np.maximum(np.minimum(x[-1],x_max),x_min)
x[-1] = (x[-1]-x_min)/(x_max-x_min)
y_pred = model.predict(x)
y = np.concatenate((abnormal_labels,
                          normal_labels), axis=0)
logits_1 = cmodel.predict(x[0],batch_size=4)
logits_2 = fmodel.predict(x[1:])

y_pred =  lamb*logits_1+(1- lamb)*logits_2
y_pred_classes = [1 if i >= threshold else 0 for i in y_pred]
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
# 计算准确率
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_classes)
# 从混淆矩阵中提取TP, TN, FP, FN
TN, FP, FN, TP = cm.ravel()
# 计算灵敏度和特异性
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(f"灵敏度（Sensitivity）: {sensitivity}")
print(f"特异性（Specificity）: {specificity}")

accuracy = accuracy_score(y, y_pred_classes)
print(f"准确率（Accuracy）: {accuracy}")
# 计算精确度
precision = precision_score(y, y_pred_classes)
print(f"精确度（Precision）: {precision}")
# 计算 F1 分数
f1 = f1_score(y, y_pred_classes)
print(f"F1 分数（F1 Score）: {f1}")
# 计算 ROC AUC 值
roc_auc = roc_auc_score(y, y_pred)
print(f"ROC AUC 值: {roc_auc}")

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
pd.DataFrame(result).to_csv('result.csv')