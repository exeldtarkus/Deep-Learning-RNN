# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:21:17 2019

@author: Edsie
"""
import sys
import sounddevice as sd
import wavio

filename= str(sys.argv[1]) + '.wav'
fs = 44100  # Sample rate
seconds = 5  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
wavio.write(filename, myrecording,fs,sampwidth=2)
