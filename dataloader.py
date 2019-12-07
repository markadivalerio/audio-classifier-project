#!/usr/bin/env python

####### All Imports #######
import warnings
warnings.filterwarnings("ignore")

import os
import random
from datetime import datetime

import librosa
from scipy.io import wavfile
import numpy as np
import pandas as pd
import sklearn as sk
import torch
from torch.utils import data
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

import IPython.display as ipd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


def main():
    ####### Setup Environment #######

    ####### Configs for Machine Learning #######
    ### Loading Test/Training Data ###
    load_urbansound_data = False # <-- Note: Urbansound8k has a shortcut for testing/debugging, only loads 1 folder (800 instead of 8000)
    # Data Source: https://urbansounddataset.weebly.com/urbansound8k.html
    load_birds_data      = False
    # Data Source: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads
    load_kaggle_data     = False
    # Data Source: https://www.kaggle.com/mmoreaux/environmental-sound-classification-50#esc50.csv
    load_kaggle_cats_dogs_data  = True
    # Data Source:  https://www.kaggle.com/c/dogs-vs-cats/data
    #               download using link https://www.kaggle.com/c/3362/download-all
    load_audioset_data   = False
    # Data Source: https://research.google.com/audioset/index.html
    # See scripts/README.md for downloading & filtering instructions.

    def extract_features(file_name):
        """
        Extracts 193 chromatographic features from sound file. 
        including: MFCC's, Chroma_StFt, Melspectrogram, Spectral Contrast, and Tonnetz
        NOTE: this extraction technique changes the time series nature of the data
        """
        features = []

        audio_data, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(audio_data))

        mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T,axis=0)
        features.extend(mfcc) # 40 = 40

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        features.extend(chroma) # 12 = 52

        mel = np.mean(librosa.feature.melspectrogram(audio_data, sr=sample_rate).T,axis=0)
        features.extend(mel) # 128 = 180

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        features.extend(contrast) # 7 = 187

    # More possible features to add
    #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X, ), sr=sample_rate).T,axis=0)
    #     spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).T, axis=0)
    #     spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).T, axis=0)
    #     rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate).T, axis=0)
    #     zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data).T, axis=0)
    #     features.extend(tonnetz) # 6 = 193
    #     features.extend(spec_cent)
    #     features.extend(spec_bw)
    #     features.extend(rolloff)
    #     features.extend(zcr)

        return np.array(features)

    from scipy.io import wavfile as wav

    def display_wav(wav_file):
        librosa_load, librosa_sampling_rate = librosa.load(wav_file)
        scipy_sampling_rate, scipy_load = wav.read(wav_file)
        print('original sample rate:',scipy_sampling_rate)
        print('converted sample rate:',librosa_sampling_rate)
        print('\n')
        print('original wav file min~max range:',np.min(scipy_load),'~',np.max(scipy_load))
        print('converted wav file min~max range:',np.min(librosa_load),'~',np.max(librosa_load))
        plt.figure(figsize=(12, 4))
        plt.plot(scipy_load)
        plt.figure(figsize=(12, 4))
        plt.plot(librosa_load)

    def load_all_wav_files(load_urbansound=False,
                           load_birds=False,
                           load_kaggle=False,
                           load_kaggle_cats_dogs=False,
                           load_audioset=False):
        '''
        Returns two numpy array
        The first is a numpy array containing each audio's numerical features - see extract_features()
        The second numpy array is the array *STRING* of the label.
        (The array indexes align up between the two arrays. data[idx] is classified as labels[idx]) 
        '''
        one_file = None
        #THIS WILL TAKE A WHILE!!!!!
        all_data = []
        all_labels = []
        all_files = []
        #UltraSound8K
        if load_urbansound:
            print("loading Ultrasound8k")
            # Data Source: https://urbansounddataset.weebly.com/urbansound8k.html
            metadata = pd.read_csv("./data/UrbanSound8K/metadata/UrbanSound8K.csv")
            for root, dirs, files in os.walk("./data/UrbanSound8K"):
                print(root, str(len(dirs)), str(len(files)), len(all_data))
    #SHORTCUT
    # This is in here for quick tests - only loads first Ultrasound8k folder (instead of all of them)
                if len(all_data) > 0: 
                    break
    #END SHORTCUT
                for idx, file in enumerate(files):
                    if file.endswith('.wav'):
                        fname = os.path.join(root, file)
                        if(len(all_data) % 100 == 0):
                            print(str(len(all_data)))
                        features = extract_features(fname)
                        label = metadata[metadata.slice_file_name == file]["class"].tolist()[0]
                        all_data.append(features)
                        all_labels.append(label)
                        one_file = fname
                        all_files.append(fname)
    #                     display_wav(fname)
    #                     break


        if load_birds:
            print("Loading birds")
            # Data Source: http://dcase.community/challenge2018/task-bird-audio-detection
            # Data Source: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads
            for root, dirs, files in os.walk("./data/warblrb10k_public_wav/train/hasbird"):
                print(root, str(len(dirs)), str(len(files)), len(all_data))
                for file in files:
                    if file.endswith('.wav'):
                        fname = os.path.join(root, file)
                        if(len(all_data) % 100 == 0):
                            print(str(len(all_data)))
                        features = extract_features(fname)
                        all_data.append(features)
                        all_labels.append("bird")
                        all_files.append(fname)


        if load_kaggle:
            print("Loading Kaggle")
            # Data Source: https://www.kaggle.com/mmoreaux/environmental-sound-classification-50#esc50.csv
            metadata = pd.read_csv("./data/environmental-sound-classification-50/esc50.csv")
            #for root, dirs, files in os.walk("./data/environmental-sound-classification-50/"):
            for file in os.listdir("./data/environmental-sound-classification-50/audio"):
                fname = "./data/environmental-sound-classification-50/audio/"+file
                if file.endswith('.wav'):
                    label = metadata[metadata.filename == file]["category"].tolist()[0]
                    animals=["cat", "chirping_birds","cow","crickets","crow","dog","frog","hen","insects","pig","rooster","sheep"]
                    if label in animals:
                        if(len(all_data) % 100 == 0):
                            print(str(len(all_data)))
                        features = extract_features(fname)
                        all_data.append(features)
                        all_labels.append(label)

        if load_kaggle_cats_dogs:
            print("Loading Kaggle cats and dogs")
            # Data Source:  https://www.kaggle.com/c/dogs-vs-cats/data
            #               download using link https://www.kaggle.com/c/3362/download-all
            metadata = pd.read_csv("./data/kaggle_cats_dogs/train_test_split.csv")
            for file in os.listdir("./data/kaggle_cats_dogs/cats_dogs"):
                fname = "./data/kaggle_cats_dogs/cats_dogs/"+file
                if file.endswith('.wav'):
                    if(len(all_data) % 100 == 0):
                        print(str(len(all_data)))
                    features = extract_features(fname)
                    all_data.append(features)
                    label = 'cat' if file.startswith('cat') else 'dog'
                    all_labels.append(label)
                    all_files.append(fname)
                    one_file = fname

        if load_audioset:
            # Data Source: https://research.google.com/audioset/index.html
            # See scripts/README.md for downloading & filtering instructions.
            print("Loading Audioset")
            metadata_b = pd.read_csv("./data/audioset/balanced_train_segments-animals.csv")
            metadata_e = pd.read_csv("./data/audioset/eval_segments-animals.csv")
            metadata_l = pd.read_csv("./data/audioset/class_labels_indices-animals.csv")
            for root, dirs, files in os.walk("./data/audioset"):
                print(root, str(len(dirs)), str(len(files)), len(all_data))
                for idx, file in enumerate(files):
                    if file.endswith('.wav'):
                        if(len(all_data) % 100 == 0):
                            print(str(len(all_data)))
                        fname = os.path.join(root, file)
                        features = extract_features(fname)
                        no_ext = file.replace(".wav", "")
                        temp = None
                        if "balanced_train_segments" in fname:
                            temp = metadata_b[metadata_b['# YTID'] == no_ext]["Unnamed: 3"].tolist()
                        elif "eval_segments" in fname:
                            temp = metadata_e[metadata_e['# YTID'] == no_ext]["Unnamed: 3"].tolist()
                        if not temp:
                            continue
                        label_code = temp[0]
                        label_temp = metadata_l[metadata.mid == label_code]["display_name"].to_list()
                        if not label_temp:
                            continue
                        label = label_temp[0]

                        all_data.append(features)
                        all_labels.append(label)
                        all_files.append(fname)


        return np.array(all_data), np.array(all_labels), all_files, one_file

    all_data, all_labels, all_files, one_file = load_all_wav_files(load_urbansound_data,
                                          load_birds_data,
                                          load_kaggle_data,
                                          load_kaggle_cats_dogs_data,
                                          load_audioset_data)
    return (all_data, all_labels, all_files, one_file)

if __name__ == "__main__":
    return_object = main()
    print("Main return:")
    print(type(return_object))
    print(return_object)