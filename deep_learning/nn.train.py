from CNN_net import SimpleVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

path= "F:/Voiceprint_Recognition_EI_CA/deep_learning/"
image_width = 64

# load data
print("------------------Begin load data------------------")
data = []
labels = []

mfcc_path = "F:/Voiceprint_Recognition_EI_CA/deep_learning/train_data"
folder_of_data = []
head_file = os.listdir(mfcc_path)
for dir_head in head_file:
    folder_of_data.append(mfcc_path + "/" + dir_head)

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

# normalization
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.3, shuffle=True,random_state=42)

#one-hot encoding ，使得离散空间取值扩展到欧式空间，某个特征取值对应某个点
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# data
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

model = SimpleVGGNet.build(width=image_width, height=image_width, depth=3,classes=len(lb.classes_))

# initial_parameter
INIT_LR = 0.01
EPOCHS = 200
BS = 8

print("------------------Start training the VGG model------------------")
#opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#运用Adam优化器
opt = Adam(lr=INIT_LR,beta_1=0.9,beta_2=0.99,epsilon=1e-8,decay=INIT_LR / EPOCHS)
#compile用于表示在配置训练时，告知训练时用的优化器、损失函数和准确率评测标准
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#     validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#     epochs=EPOCHS)

#Start training
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=BS)


# validation
print("------Validation step------")
predictions = model.predict(testX, batch_size=4)
print (accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1))) 
print(f1_score(testY.argmax(axis=1), predictions.argmax(axis=1), labels=None, pos_label=1, average='micro',sample_weight=None))
print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))

# plot the results
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.xlim(0,200)
plt.ylim(0,5)
plt.plot(N, H.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(path+"output/cnn_plot.png")

print("------save the model------")
model.save(path+"/output/cnn.model")
f = open(path+"output/cnn_lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()