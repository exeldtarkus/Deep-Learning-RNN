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

def build_predictions(audio_dir):
#    y_true = []
    y_pred = []
    fn_prob = {}
    
    print("Extracting features from audio")
    _min, _max = float('inf'), -float('inf')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        #label = fn2class[fn]
        #c= classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, 
                     nfft=config.nfft)
            _min = min(np.amin(x), _min)
            _max = max(np.amax(x), _max)
            x = (x - _min) / (_max - _min)
           
            x = np.expand_dims(x, axis=0)
            
            y_hat = model.predict(x)
            
            y_prob.append(y_hat)
#            y_true.append(c)
            y_pred.append(np.argmax(y_hat))

        fn_prob[fn] = np.mean(y_prob, axis=0).flatten(0)
        #print("y_prob : ", y_prob)

    return y_pred, fn_prob
 
file= pd.read_excel('cobaprediksi.xlsx')
p_path = os.path.join('pickles', 'time.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)    
#rate, datatest = wavfile.read('data_training/audio_test.wav')
y_pred, fn_prob = build_predictions("data_training")
#acc_score = accuracy_score(y_pred=y_pred)

#y_probs = []
#for i, row in file.iterrows():
#    y_prob = fn_prob[row.fname]
#    y_probs.append(y_prob)
#    for c, p in zip(classes, y_prob):
#        file.at[i, c] = p

#y_pred = [classes[np.argmax(y)]for y in y_probs]
file["saya"] = y_pred

file.to_excel("prediksi2.xlsx", index = False)


