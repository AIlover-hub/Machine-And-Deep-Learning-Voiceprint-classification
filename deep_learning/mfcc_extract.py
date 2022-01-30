from pydub import AudioSegment
from scipy.io.wavfile import read
import os
import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt

path = "F:/Voiceprint_Recognition_EI_CA"
train_data_path = "F:/Voiceprint_Recognition_EI_CA/machine_learning/train_data/"
test_data_path = "F:/Voiceprint_Recognition_EI_CA/machine_learning/test_data/"

#Construct the MFCC image for deep learning method

count =0
#提取MFCC特征
def get_train_features(song,directory,plotname):
    global count
    y,_ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10,4))                                                                            #输出图像1000*400像素
    librosa.display.specshow(mfcc,x_axis='time',y_axis='mel')
    plt.tight_layout()
    plt.savefig( path+"/deep_learning/train_data/"+directory + "/" + plotname.split( '.' )[0] + '.png' )
    plt.close( 'all' )
    count = count+1

def get_test_features(song, directory, plotname):
    global count
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))  # 输出图像1000*400像素
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.savefig(path + "/deep_learning/test_data/" + directory + "/" + plotname.split('.')[0] + '.png')
    plt.close('all')
    count = count + 1

    return 0
#遍历生成MFCC特征图
train_files = os.listdir(train_data_path)
test_files = os.listdir(test_data_path)
for train_file in train_files:
    dataset_of_wavs = []
    for (_,_,filenames) in os.walk(train_data_path + train_file):
        dataset_of_wavs.extend(filenames)
        break
    for dataset_of_wav in dataset_of_wavs:
        get_train_features(train_data_path + train_file + "/" + dataset_of_wav, train_file, dataset_of_wav)

for test_file in test_files:
    dataset_of_wavs = []
    for (_, _, filenames) in os.walk(test_data_path + test_file):
        dataset_of_wavs.extend(filenames)
        break
    for dataset_of_wav in dataset_of_wavs:
        get_test_features(test_data_path + test_file + "/" + dataset_of_wav, test_file, dataset_of_wav)

