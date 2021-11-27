#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 10:55:00 2021

@author: serapf
"""
import numpy as np
import random
import os
import torch #
import torch.nn as nn #tools for neuronal networks
import torch.nn.functional as F #Loss function, activation functions ,to implement nn
import torch.optim as optim #Gradients descent
import torch.autograd as autograd
from torch.autograd import Variable #Converto to torchTensor to a variable that contains the tensor and the gradient

class Network (nn.Module):
    
    def __init__(self, input_size, nb_class): #this, number of signals (states of enviroument), number of actions (Q)
        super(Network, self).__init__() #Initialize the current object as a NN, it is complement of inhertance
        
        #Store the parameters as atributes of the Network
        self.input_size = input_size
        self.nb_class = nb_class
        
        #Creation and full conections of layers
        self.fc1 = nn.Linear(input_size, 120) #Full Conection, input size - hiden layer = 30
        self.fc2 = nn.Linear(120, 70) #Full Conection, input size - hiden layer = 30
        self.fc3 = nn.Linear(70, nb_class) #Full Conection, hiden layer - output layer
        
    def forward(self, data): #state input of neuronal network
        x = F.relu(self.fc1(data)) # x - hidden neurons, F contains all the functions for neurons, relu - rectifier function
        x = F.relu(self.fc2(x)) # x - hidden neurons, F contains all the functions for neurons, relu - rectifier function
        pred = self.fc3(x) #using the output layer, input the results of x and get q-values, with no activation because they are output it will be treat with softmax
        return pred
        
class Gakki_NN():
    
    def __init__(self, input_size, nb_class):
        self.model = Network(input_size, nb_class) #Initializing model from Network
            
        #Using adaptive model
        #adam takes de paramenter of a model nn.Model to edit them, lr = larning rate (give the opportunity to the AI to learn (exploration))
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
    def process(self, batch, instrument_prob):
        
        chunkdata = torch.Tensor(batch).float().unsqueeze(0)
        target_prob = torch.Tensor(instrument_prob).float().unsqueeze(0)
        
        pred = self.model(Variable(chunkdata, volatile = True))
        
        probs = F.softmax(pred)
        
        td_loss = F.smooth_l1_loss(probs, target_prob)
        
        self.optimizer.zero_grad() #reinitilize at each iteration of the loop
        td_loss.backward(retain_graph=True) #will apply adam, retain graph free memory
        self.optimizer.step() #update the wights
        
        return td_loss.item(), probs
    
    def predict(self, data):
        chunkdata = torch.Tensor(data).float().unsqueeze(0)
        pred = self.model(Variable(chunkdata, volatile = True))*100
        probs = F.softmax(pred)
        
        return probs
    
    def save(self):
        #{model - object,
        # optimizer - object}
        #, 'file name'
        
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            #functions to load the modle and optimizer into "self" object
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
    
        
        
    