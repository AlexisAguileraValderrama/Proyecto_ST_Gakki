#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 10:55:00 2021

@author: serapf
"""
from sys import getsizeof
from scipy.fft import fft, ifft

from os import walk

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics


import torch
import torchvision
import torchvision.transforms as transforms

from red_neuronal import Gakki_NN

# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

path_to_samples='./Samples'
path_to_datasets = './Datasets'

segundos = 0.9
umbral = 10
sample_rate = 48000
    
chunk = int(segundos * sample_rate)

tam = int(segundos * sample_rate / 2)
    
modelo = Gakki_NN(tam, 8)

ins = {
        0    : "Bass",
        1    : "Flute",
        2    : "Guitar",
        3    : "Kalimba",
        4    : "Piano",
        5    : "Saxophone",
        6    : "Shamisen",
        7    : "Violin",
}
 
def getSamples(batch ,file, num_samples, instrument_index):
    
    transformada = []
    
    collected_samples = 0
    
    line = file.readline()
                
    while line != "" :
        if line != "\n":
            transformada.append(float(line))
        else:
            batch.append([np.array(transformada),instrument_index])
            transformada = []
            collected_samples = collected_samples + 1
            if collected_samples == num_samples:
                break
        line = file.readline()
    
def CreateDataSet():
    

    filenames = sorted(next(walk(path_to_samples), (None, None, []))[2])
    
    """
        0 - Bass
        1 - Flute
        2 - Guitar
        3 - Kalimba
        4 - Piano
        5 - Shamisen
        6 - Violin
    """
    
    for file_name in filenames:
        path = path_to_samples + "/" + file_name
        
        samplerate, data = wavfile.read(path)
        
        mono = data[:,0]
        
        mono = np.delete(mono, np.where(np.abs(mono)<umbral))
        
        print("Mono size ",len(mono))
        
        i = 0
        
        print("Procesando ",file_name)
        
        with open(path_to_datasets + "/" + file_name.split(".",1)[0] +".txt" ,"w") as f:
        
            while True:
            
                    samples = mono[chunk*i:chunk*(i + 1)]
                
                    if len(samples) < chunk:
                        break
                
                    transformada = np.abs(fft(samples) * 2 / (256 * chunk))
                
                    transformada = transformada[int(chunk/2):len(transformada)]
                    
                    for value in transformada:
                        f.write(str(value) + "\n")
                        
                    f.write("\n")
                
                    i = i + 1
    
def train():
    
    batch = []
    
    samples_per_instrument = 30
    
    instrument_dict = {
            "Bass"     : 0,
            "Flute"    : 1,
            "Guitar"   : 2,
            "Kalimba"  : 3,
            "Piano"    : 4,
            "Saxophone": 5,
            "Shamisen" : 6,
            "Violin"   : 7,
        }
    
    prob_instrument = {
            0    : [1.0,0,0,0,0,0,0,0], ##Bajo
            1    : [0,1.0,0,0,0,0,0,0], ##Flauta
            2    : [0,0,1.0,0,0,0,0,0], ##Guitarra
            3    : [0,0,0,1.0,0,0,0,0], ##Kalimba
            4    : [0,0,0,0,1.0,0,0,0], ##Piano
            5    : [0,0,0,0,0,1.0,0,0], ##Saxophone
            6    : [0,0,0,0,0,0,1.0,0], ##Shamisen
            7    : [0,0,0,0,0,0,0,1.0], ##Violin
        }
    
    err = [0,0,0,0,0,0,0,0]
    
    tam = int(segundos * sample_rate / 2)
    
    modelo = Gakki_NN(tam, 8)
    
    i = 1
    
    with open("./Datasets/Bass.txt") as bass_file,\
         open("./Datasets/Flute.txt") as flute_file,\
         open("./Datasets/Guitar.txt") as guitar_file,\
         open("./Datasets/Kalimba.txt") as kalimba_file,\
         open("./Datasets/Piano.txt") as piano_file,\
         open("./Datasets/Saxophone.txt") as saxophone_file,\
         open("./Datasets/Shamisen.txt") as shamisen_file,\
         open("./Datasets/Violin.txt") as violin_file:
             
            while True:
                 
                getSamples(batch, bass_file,     samples_per_instrument,instrument_dict["Bass"])
                getSamples(batch, flute_file,    samples_per_instrument,instrument_dict["Flute"])
                getSamples(batch, guitar_file,   samples_per_instrument,instrument_dict["Guitar"])
                getSamples(batch, kalimba_file,  samples_per_instrument,instrument_dict["Kalimba"])
                getSamples(batch, piano_file,    samples_per_instrument,instrument_dict["Piano"])
                getSamples(batch, saxophone_file, samples_per_instrument,instrument_dict["Saxophone"])
                getSamples(batch, shamisen_file, samples_per_instrument,instrument_dict["Shamisen"])
                getSamples(batch, violin_file,   samples_per_instrument,instrument_dict["Violin"])
                
                if i == 90:
                    break;
                
                random.shuffle(batch)
                
                accumulated_loss = []
                
                for element in batch:
                    loss,prob = modelo.process(element[0], prob_instrument[element[1]])
                    accumulated_loss.append(loss)
                    
                    prob = prob.detach().numpy()[0].tolist()
                    
                    max_val = max(prob)
    
                    index = prob.index(max_val)
                    
                    if index != element[1]:
                        err[element[1]] = err[element[1]] + 1
                    
                        
                
                print("-----------------------------------------------")
                print("Batch no. ", i)
                print("Current samples num ", samples_per_instrument * i)
                print("Batch Size ", len(batch)/8)
                print("Batch loss ", statistics.mean(accumulated_loss))
                print("Errores ", err)
                print("-----------------------------------------------\n")
                batch = []
                i = i + 1
    return modelo
        
    
def load_model():
    
    modelo.load()

def Listen():
    
    record()
    
    samplerate, data = wavfile.read("recording1.wav")
    
    mono = data[:,0]
    
    samples = mono[0:chunk]
    
    transformada = np.abs(fft(samples) * 2 / (256 * chunk))
    
    transformada = transformada[int(chunk/2):len(transformada)]
        
    pred = modelo.predict(transformada).detach().numpy()[0].tolist()
        
    max_val = max(pred)
    
    index = pred.index(max_val)
    
    print(ins[index])
        
    return ins[index]
    
        
def record():
    # Sampling frequency
    freq = sample_rate

    # Recording duration
    duration = segundos + 0.2
    
    # Start recorder with the given values of 
# duration and sample frequency
    recording = sd.rec(int(duration * freq), 
    samplerate=freq, channels=2)
    
    sd.wait()
    
    wv.write("recording1.wav", recording, freq, sampwidth=2)
        

    
    
    
