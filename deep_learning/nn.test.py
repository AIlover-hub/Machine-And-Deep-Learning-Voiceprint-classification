from keras.models import load_model
import argparse
import pickle
import cv2
import os
import numpy as np

image_width = 64

# ===========================load testing data===========================#
print("--------------------loading test data--------------------")
data = []
labels = []

mfcc_test_path = "F:/Voiceprint_Recognition_EI_CA/deep_learning/test_data"
folder_of_data = []
head_file = os.listdir(mfcc_test_path)
for dir_head in head_file:
    folder_of_data.append(mfcc_test_path + "/" + dir_head)

print("----------------Starting process image----------------")
count = 0
for location in folder_of_data:
    data_plots = []
    for(_,_,filenames) in os.walk(location):
        data_plots.extend(filenames)
        break
    for data_plot in data_plots:
        image = cv2.imread(location + "/" + data_plot)
        image = cv2.resize(image, (image_width, image_width))
        data.append(image)
        labels.append(count)
    count += 1
# ==============================================================#

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

save_path = "F:/Voiceprint_Recognition_EI_CA/deep_learning/"
print("------load model and label------")
model = load_model(save_path + "output/cnn.model")
lb = pickle.loads(open(save_path + "output/cnn_lb.pickle", "rb").read())

predict_lable=[]
# predict
for data_item in data:
    data_item = data_item.reshape((1, data_item.shape[0], data_item.shape[1],data_item.shape[2]))
    preds = model.predict(data_item)

    # get the prediction and the corresponding label
    i = preds.argmax(axis=1)[0]       #axis=1,在行中比较，取出最大的列向量
    label = lb.classes_[i]
    predict_lable.append(label)

# calculate the accuracy
count_right = 0
i = 0
while i<85:
    if i <5:
        if predict_lable[i]==0:
            count_right+=1
    if i>4 and i<10:
        if predict_lable[i]==1:
            count_right+=1
    if i>9 and i<15:
        if predict_lable[i]==2:
            count_right+=1
    if i>14 and i<20:
        if predict_lable[i]==3:
            count_right+=1
    if i>19 and i<25:
        if predict_lable[i]==4:
            count_right+=1
    if i>24 and i<30:
        if predict_lable[i]==5:
            count_right+=1
    if i>29 and i<35:
        if predict_lable[i]==6:
            count_right+=1
    if i>34 and i<40:
        if predict_lable[i]==7:
            count_right+=1
    if i>39 and i<45:
        if predict_lable[i]==8:
            count_right+=1
    if i>44 and i<50:
        if predict_lable[i]==9:
            count_right+=1
    if i>49 and i<55:
        if predict_lable[i]==10:
            count_right+=1
    if i>54 and i<60:
        if predict_lable[i]==11:
            count_right+=1
    if i>59 and i<65:
        if predict_lable[i]==12:
            count_right+=1
    if i>64 and i<70:
        if predict_lable[i]==13:
            count_right+=1
    if i>69 and i<75:
        if predict_lable[i]==14:
            count_right+=1
    if i>74 and i<80:
        if predict_lable[i]==15:
            count_right+=1
    if i>79:
        if predict_lable[i]==16:
            count_right+=1
    i +=1

print('预测的准确度'+str(count_right/85))
