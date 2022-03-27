# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 02:11:57 2019

@author: Edsie
"""

import os
import numpy as np
import wave
import struct
import pandas as pd
from scipy.io import wavfile
import noisereduce as nr
import wavio


def audio_list(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.wav' in file:
                files.append(os.path.join(r, file))
    return files

def wav_to_fft(wav_file):
    num_samples = 48000
    wav_file = wave.open(wav_file, 'r')
    data = wav_file.readframes(num_samples)
    wav_file.close()
    data = struct.unpack('{n}h'.format(n=num_samples), data)
    data = np.array(data)
    data_fft = np.fft.fft(data)
    return np.abs(data_fft)

def load_labels(csv_file):
    data = pd.read_csv(csv_file)
    labels= data['label'].values.tolist()
    return labels

def noise_reduce(audio_file,noise_file):
    rate, data = wavfile.read(audio_file)
    data = data / 44100
    # select section of data that is noise
    noise_rate, noisy_part = wavfile.read(noise_file)
    noisy_part=noisy_part/44100
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False)
    return reduced_noise