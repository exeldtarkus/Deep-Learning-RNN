# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:06:03 2019

@author: user
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from cfg import Config

def build_predictions(audio_dir):
#    y_true = []
    y_pred = []
    fn_prob = {}
    y_prob = []
#    nilai_prediksi = []
    nfilt = 26
    nfeat = 13
    nfft = 512
    rate = 16000
    step = int(rate/10)
    
    print("Extracting features from audio")
    _min, _max = float('inf'), -float('inf')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        #label = fn2class[fn]
        #print("Label : ", label)
        #c= classes.index(label)
        #print('Classes :', c)
        
        
        for i in range(0, wav.shape[0]-step, step):
            sample = wav[i:i+step]
            x = mfcc(sample, rate, numcep=nfeat, nfilt=nfilt, 
                     nfft=nfft)
            _min = min(np.amin(x), _min)
            _max = max(np.amax(x), _max)
            x = (x - _min) / (_max - _min)
           
            x = np.expand_dims(x, axis=0)
            
            y_hat = model.predict(x)
            
            y_prob.append(y_hat)
#            y_true.append(c)
            y_pred.append(np.argmax(y_hat))

        fn_prob[fn] = np.mean(y_prob, axis=0).flatten(0)
#        nilai_prediksi = fn_prob[fn]
#        print("ini adalah fn_prob : ",fn_prob)
#        print()
#        print(y_hat)

    return y_pred, fn_prob
 
#file= pd.read_excel('datafitur-nama-class1.xlsx')
#classes = list(np.unique(file.label))
#classes = ['bad','good']




#fn2class = dict(zip(file.fname, file.label))

#p_path = os.path.join('pickles', 'time.p')
#with open(p_path, 'rb') as handle:
#    config = pickle.load(handle)


#model_path = "./models/time.h5"
model = load_model("time.h5")
print("Model Loaded.")
print(model)    
#rate, datatest = wavfile.read('data_training/audio_test.wav')

y_pred, fn_prob = build_predictions("data_training")
for i in fn_prob:
    value1 = (fn_prob[i][0])
    value2 = (fn_prob[i][1])

if value1 < value2:
    print('Telur Bagus')
elif value1 > value2:
    print('Telur Bisae')
else:
    print("File Tidak Terbaca...!!!")
#acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

#y_probs = []
#for i, row in file.iterrows():
#    y_prob = fn_prob[row.fname]
#    y_probs.append(y_prob)
#    for c, p in zip(classes, y_prob):
#        file.at[i, c] = p

#y_pred = [classes[np.argmax(y)]for y in y_prob]

#file["y_pred"] = y_pred
#file.to_excel("prediksi4.xlsx", index = False)


