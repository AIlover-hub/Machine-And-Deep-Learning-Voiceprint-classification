import os
import librosa
import librosa.feature
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

print("------------------starting------------------")

#extract the MFCC features
def get_features(song):
    y,_ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)               #提取时域信号y的梅尔频谱
    mfcc /= np.amax(np.absolute(mfcc))       #归一化，找绝对值数组最大的数，除
    return np.ndarray.flatten(mfcc)             #扁平化数组

print("#=============construct the training dataset=============#")

#=============construct the training dataset=============#
X = []
y = []

train_path = "F:/Voiceprint_Recognition_EI_CA/machine_learning/train_data"
test_path = "F:/Voiceprint_Recognition_EI_CA/machine_learning/test_data"

count_train = 0
train_files = os.listdir(train_path)
target_names = []
for train_file in train_files:
    target_names.append(train_file)
    dataset_of_wavs = []
    for (_,_,filenames) in os.walk(train_path + "/" + train_file):
        dataset_of_wavs.extend(filenames)
        break
    for dataset_of_wav in dataset_of_wavs:
        X.append(get_features(train_path + "/" + train_file + "/" + dataset_of_wav))
        y.append(count_train)
    count_train += 1

#========================================================================#
print("#=============training dataset is done=============#")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print("#=============Training Action=============#")

#training
clf = LinearSVC(random_state=0, tol=1e-5)                 #tol:公差停止标准；random_state:在随机数据混洗时使用的伪随机数生成器的种子
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print("#=============Getting the training result=============#")

#get the training result
print(accuracy_score(y_test, predicted))
print(f1_score(y_test, predicted, labels=None, pos_label=1, average='micro',sample_weight=None))
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted, target_names=target_names))

print("#=============construct the testing dataset=============#")
#=========construct the testing dataset=========#
X_predict=[]
y_predict=[]

count_test = 0
test_files = os.listdir(test_path)
for test_file in test_files:
    dataset_of_wavs = []
    for (_, _, filenames) in os.walk(test_path + "/" + test_file):
        dataset_of_wavs.extend(filenames)
        break
    for dataset_of_wav in dataset_of_wavs:
        X_predict.append(get_features(test_path + "/" + test_file + "/" + dataset_of_wav))
        y_predict.append(count_test)
    count_test += 1
    if count_test % 10 ==0:
        print("10 of testing data is setting done")
#==============================================================#

print("#=============get the testing result=============#")
#get the testing result
predicted_predict = clf.predict(X_predict)
print(accuracy_score(y_predict, predicted_predict))
print(f1_score(y_predict, predicted_predict, labels=None, pos_label=1, average='micro', sample_weight=None))
print(confusion_matrix(y_predict, predicted_predict))
print(classification_report(y_predict, predicted_predict, target_names=target_names))
