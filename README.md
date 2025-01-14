train_feature.py:train a Clinical feature model
input data format is belike
Radiology number, grouping, clinical diagnosis, gender, smoking history, drinking history, hypertension history, diabetes history, number of radiotherapy sessions, pathological binary classification, initial radiotherapy T staging, initial radiotherapy N staging, MRI necrosis depth, binary classification of necrosis morphology, whether there are positive lymph nodes at the time of necrosis, age, weight kg at the time of necrosis, BMI at the time of necrosis, RBC at the time of necrosis, HGB at the time of necrosis, WBC at the time of necrosis, absolute value of neutrophils at the time of necrosis, percentage of neutrophils at the time of necrosis, absolute value of lymphocytes at the time of necrosis, percentage of lymphocytes at the time of necrosis, absolute value of monocytes at the time of necrosis, percentage of monocytes at the time of necrosis, platelet count at the time of necrosis.
Count PLT, LDH at necrosis, TP at necrosis, ALB at necrosis, GLOB at necrosis, UREA at necrosis, CRE at necrosis, CRP at necrosis, CK at necrosis, SAA at necrosis, GLU at necrosis, EBVDNA at necrosis.
10269914,train,1,0,1,1,0,0,1,0,3,2,2,0,0,42,54,15.7779401,3.23,90,7.04,5.5,78.6,0.6,8.8,0.8,11.1,489,115.2,65.16,32.5,32.66,3.7,48.1,42.26,53,82.8,5.1,2220

train_swinT.py:train an MRI classfier model using swinT
download pretrain model weights by https://github.com/innat/VideoSwin

vaild.py:Validate the  model joinly MRI model and feature model
