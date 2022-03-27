# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:36:46 2019

@author: Edsie
"""
import wavio
from function import audio_list,noise_reduce


path="sample_001.wav"
noise_file="NoiseFile.wav"
save_path="data_clean"

files=audio_list(path)

for audio_file in files:
    audio_signal=noise_reduce(audio_file,noise_file)
    file=audio_file.split('\\')
    save_file= save_path+"\\" +str(file[1])
    print (save_file)
    wavio.write(save_file, audio_signal,44100,sampwidth=2)


 