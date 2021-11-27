#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:49:57 2021

@author: serapf
"""

import tkinter
from tkinter import Label,Button
from PIL import ImageTk, Image

from threading import Thread

import Core

Core.load_model()

window = tkinter.Tk()

comenzar = False

def Comenzar():
    global comenzar
    if comenzar == False:
        thread = Thread(target = Ejecutar)
        thread.start()
        comenzar = True
    else:
        print("Ya esta corriendo")

def Ejecutar():
    my_label.config(text = "Escuchando...")
    while True:
        respuesta = Core.Listen()
        
        imagen = ImageTk.PhotoImage(Image.open(respuesta + ".jpg").resize((500,300)))
        my_image.config(image = imagen)
        my_label.config(text = respuesta)

global my_image
imagen = ImageTk.PhotoImage(Image.open("title.jpg").resize((500,300)))
my_image = Label(image = imagen)
my_image.pack()

global my_label
my_label = Label(window, text="Da click para comenzar a predecir!!!")
my_label.pack(pady=10)

boton = Button(text="Comenzar!",command = Comenzar)
boton.pack()

global my_label_disclaim
my_label_disclaim = Label(window, text="AVISO: Para mejor detección del sonido se recomienda usar el audio directo de la computadora\n" +
                                       "Para lograr esto desconecte todos los microfonos externos y habilite la entrada de audio del sistema\n"+
                                       " y ponga el audio en youtube, spotify, soundcloud o el software que guste.")
my_label_disclaim.pack(pady=10)

global my_label_info
my_label_info = Label(window, text="El proyecto Gakki (楽器) es capaz de detectar hasta 8 instrumentos disponbles" + 
                                      " tales como: \n Bajo, guitarra, flauta, kalimba, shamisen, violin, piano y saxofón\n" +
                                      "El proyecto fue desarollado por Aguilera Valderrama Alexis Fernando (Serapf) para \n"+
                                      " el proyecto final de sistema de comunicaciones del grupo 5 del semestre 2022-1")
my_label_info.pack(pady=10)

window.geometry("800x500")
window.title('Gakki (楽器)')

window.mainloop()
    
    