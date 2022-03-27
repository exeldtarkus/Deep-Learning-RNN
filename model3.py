# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:37:22 2019
model coba untuk traning data

@author: user
"""

import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import LSTM, Flatten, Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    
def build_rand_feat():
    print()
    print("Load Wavfile data Traning...........")

    #fungsi untuk pemisahan x dan y
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        files = np.random.choice(file[file.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+files)
        label = file.at[files, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(label))
    
    X, y = np.array(X), np.array(y)
    #ambil data X
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    
    #ambil data y
    y = to_categorical(y, num_classes=10)
    config.data = (X, y)
    
#   with open(config.p_path, 'wb') as handle:
#       pickle.dump(config, handle, protocol=2)
    return X, y

#class Config:
#    def __init__ (self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
#        self.mode = mode
#        self.nfilt = nfilt
#        self.nfeat = nfeat
#        self.nfft = nfft
#        self.rate = rate
#        self.step = int(rate/10)
#        self.model_path = os.path.join('models', mode + '.model')
#        self.p_path = os.path.join('pickles', mode + '.p')

def get_rnn_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.summary()
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
    return model

def gambar_diagram():
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()
    

file = pd.read_excel('datafitur-nama-class.xlsx')
file.set_index('fname', inplace=True)

for f in file.index:
    rate, signal = wavfile.read('clean/'+f)
    file.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(file.label))
class_dist = file.groupby(['label'])['length'].mean()

n_samples = 2 * int(file['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)


#gambar_diagram()
config = Config(mode='time')
X, y = build_rand_feat()
y_flat = np.argmax(y, axis=1)
input_shape = (X.shape[1], X.shape[2])
model = get_rnn_model()

#model_path = './models'
model_path = os.path.join('models', config.mode + '.h5')

class_weight = compute_class_weight('balanced', 
                                    np.unique(y_flat), 
                                    y_flat)


checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                             mode='max', save_best_only=True, 
                             save_weights_only=True,
                             period=1)

model.fit(X, y, epochs=100, batch_size=32, shuffle=True, validation_split=0.1,
          callbacks=[checkpoint])

#model.fit(X, y, epochs=10, batch_size=32)
model=model.save_weights(model_path)
print("Model Saved.!")
