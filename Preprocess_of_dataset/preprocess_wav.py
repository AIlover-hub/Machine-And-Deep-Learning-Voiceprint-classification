#transfer .flac to .wav and save in test folder
import os
from os import walk
from pydub import AudioSegment
parameters = None

dataset_path = "F:/librispeech_finetuning/"
path_globle="F:/Voiceprint_Recognition_EI_CA"
folder_of_data = []
name_of_data = []
location_of_data = []
head_file = os.listdir(dataset_path)
for dir_head in head_file:
    folder_of_data.append(dataset_path  + dir_head)

for location in folder_of_data:
    dataset_of_wavs = []
    data = []
    for (_, _, filenames) in walk(location):
        dataset_of_wavs.extend(filenames)
        break
    for dataset_of_wav in dataset_of_wavs:
        song = AudioSegment.from_file(location + "/" + dataset_of_wav)
        song.export("F:/Voiceprint_Recognition_EI_CA/split_dataset" + location[-3:] + "/" + location[-3:] + "_" + dataset_of_wav[12:16] + ".wav", format="wav")

#Add the file and select the sound file which is larger than 10min02s
dataset_train_path = "F:/Voiceprint_Recognition_EI_CA/split_dataset/"
folder_of_data = []
name_of_data = []
location_of_data = []
head_file = os.listdir(dataset_train_path)
for dir_head in head_file:
    folder_of_data.append(dataset_train_path  + dir_head)

a = []
count = 1
for location in folder_of_data:
    dataset_of_wavs = []
    sum_music = 0
    for (_, _, filenames) in walk(location):
        dataset_of_wavs.extend(filenames)
        break
    for dataset_of_wav in dataset_of_wavs:
        input_music = AudioSegment.from_wav(location + "/" + dataset_of_wav)
        sum_music = sum_music + input_music
    if len(sum_music) >= 660000:
        sum_music.export(path_globle + "/dataset/" + str(count).rjust(2,'0') + "_output_music.wav", format="wav")
        count += 1