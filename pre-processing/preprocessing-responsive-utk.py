import os
import time
import threading
import multiprocessing
import math
from pylab import *
import PIL.Image as im
import csv
import sys
import pandas as pd
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from string import Template
import string

#--La funzione scelta consente di poter scegliere con quale tipologia di ragnatela avviare il pre-processing--#
def scelta():
    print("E' possibile scegliere tra 4 configurazione: ")
    print("   1 - 4 cerchi 4 settori Variante 4")
    print("   2 - 4 cerchi 4 settori Variante 2")
    print("   3 - 4 cerchi 3 settori ")
    print("   4 - 5 cerchi 4 settori")
    numero = int(input("Quale configurazione scegli: "))
    if numero > 4 or numero < 1:
        print("Parametro errato.")
        exit(0)
    return numero

def distanza(x1, y1, x2, y2):
    x12 = (x2 - x1) * (x2 - x1)
    y12 = (y2 - y1) * (y2 - y1)
    xy = x12 + y12
    dist = math.sqrt(xy)
    return dist

#Alla funzione aggiungi vengono passati due ulteriori parametri cerchi e scelta_config
#cerchi: è necessaria per specificare in quante parti deve essere diviso il raggio per formare la ragatela
#scelta_config: è necessaria per specificare con quale configurazione bisogna creare la ragnatela
def aggiungi(xcentro, ycentro, rax, xpunto, ypunto, distNaso, coeff, cerchi, scelta_config):

    settore = np.zeros(3) #cerchio, quadrante, fetta
    # distNaso =  distanza dal naso
    a = 0  # a = raggioStart

    conf = [
        ['4C_4S_var4', 4 * rax / 10, 7 * rax / 10, 9 * rax / 10],
        ['4C_4S_var2', 8 * rax / 15, 12 * rax / 15, 14 * rax / 15],
        ['4C_3S_inv', rax/4, rax/2, 3 * rax / 4],
        ['5C_4S_inv', rax / 5, 2 * rax / 5, 3 * rax / 5, 4 * rax / 5]]

    #---------------------------- anello ----------------------------#
    #definisce in quale anello ci troviamo analizzando la distanza dal naso
    #non utilizza più i parametri b1,b2, etc ma li prende dalla lista conf, l'indice viene passato alla funzione quando chiamata
    if( distNaso > a and distNaso <= conf[scelta_config][1]):
        settore[0] = 1
    elif(distNaso > conf[scelta_config][1] and distNaso <= conf[scelta_config][2]):
        settore[0] = 2
    elif(distNaso > conf[scelta_config][2] and distNaso <= conf[scelta_config][3]):
        settore[0] = 3
    elif (cerchi == 5):     #è stato aggiunto un if per prevedere il quinto cerchio, non previsto nell'algoritmo fornito
        if(distNaso > conf[scelta_config][3] and distNaso <= conf[scelta_config][4]):
            settore[0] = 4
        else:
            settore[0] = 5
    else:
        settore [0] = 4
    #---------------------------- quadrante --------------------------#
    #definisce in quale quadrante ci troviamo in base alle coordinate del punto
    if (xpunto <= xcentro and y <= ycentro):
        # il punto appartiene al quadrante in alto a sinistra
        settore[1] = 2
    elif (x <= xnose and y >= ynose):
        # il punto appartiene al quadrante in basso a sinistra
        settore[1] = 3
    elif (x >= xnose and y <= ynose):
        # il punto appartiene al quadrante in alto a destra
        settore[1] = 1
    else:
        # il punto appartiene al quadrante in basso a destra
        settore[1] = 4
#------------------------- Fetta del quadrante -----------------------#
    b = 90  / fetteQ      #grado Stop
    i = 1                 #in quale fetta cade il punto. i = [1, fette]

    radang_a = 0                    # radiante Start
    radang_b = math.radians(b)      # radiante Stop
    tng_a = math.tan(radang_a)
    tng_b = math.tan(radang_b)

    #fetta
    while(settore[2] == 0 and b < 90):
        if coeff > tng_a and coeff <= tng_b:
            settore[2] = i
        b = b + (90  / fetteQ)
        radang_b = math.radians(b)  # radiante Stop
        tng_a = tng_b
        tng_b = math.tan(radang_b)
        i = i+1

    if xpunto == xnose:
        settore[2] = 1

    if settore[2] == 0:
        settore[2] = fetteQ

    if settore[1] == 1 or settore[1] == 3:
        indice = int(fette * (settore[0]-1) + fetteQ * (settore[1] - 1) + abs(settore[2] - 4) - 1)

    else:
        indice = int(fette * (settore[0] - 1) + fetteQ * (settore[1] - 1) + settore[2] - 1)

    try:
        if xnose != xpunto or ynose != ypunto:           #il naso non ha settore
            volto[indice] = int(volto[indice] + 1)       #aggiunge 1 al contatore del settore contenente il landmark
    except:
        print("Errore, non è stato possibile addizionare il landmark al settore della ragnatela.")
        print("L'indice del landmark non addizionato  è:  " + str(indice))

#la funzione reponsive_resize riceve in input l'immagine del datataset e ne individua la presenza di un volto
#si è notato che alcuni volti a bassa risoluzione non vengono rilevati, si prova così ad effettuare l'individuazione
#a diverse risoluzioni fino ad un massimo di 2048, dopo tale risoluzione l'immagine o non contiene un volto o il
#volto presenta elementi di disturbo che ne impediscono l'individuazione
def responsive_resize(image):
    image = imutils.resize(image, width=256)    #ridimensiona l'immagine
    rects = detector(image, 1)                  #rileva la presenza di volti all'interno dell'immagine
    if len(rects) == 0:
        image = imutils.resize(image, width=512)
        rects = detector(image, 1)
        if len(rects) == 0:
            image = imutils.resize(image, width=1024)
            rects = detector(image, 1)
            if len(rects) == 0:
                image = imutils.resize(image, width=2048)
                rects = detector(image, 1)
                if len(rects) == 0:
                    print("Immagine non riconosciuta: " + nomeimmagine)
    return rects, image

def stampa_volto(x, y, w, h, shape, image):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 153, 0), 2)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow('image', image)
    cv2.waitKey(0)

#la funzione write_list_to_file permette di salvare all'interno di un file csv il dataset trasformato,
#ovvero da un input di sole immagini ed etichette si avrà un nuovo dataset con un array contentente la ragnatela
#e le rispettive altre features note dal precedente dataset.
def write_list_to_file(guest_list, filename):
    contatore = 0
    with open(filename, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ';')
        for entries in guest_list:
            csvwriter.writerow(entries) #transf_blocks-->writerow
            contatore = contatore +1
        print("Scritti correttamente " + str(contatore) + " elementi. Il numero di elementi scartati  e': " + str(size_dataset - contatore))
    csvfile.close()

detector = dlib.get_frontal_face_detector()                                 #individua il volto il un immagine
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   #individua i 68 landmark sul volto


# indice configurazione per reperire i dati del raggio nella lista configurazioni in aggiungi, tale parametro va passato ad ogni chiamata di aggiungi
# 0 - C4_4S_var4
# 1 - C4_4S_var2
# 2 - C4_3S_inv
# 3 - C5_4S_inv
numero_scelta = int(scelta())
if numero_scelta==1:
    anelli = 4
    fetteQ = 4   # fette per quadrante
    variante = 'var4'
elif numero_scelta == 2:
    anelli = 4
    fetteQ = 4  # fette per quadrante
    variante = 'var2'
elif numero_scelta == 3:
    anelli = 4
    fetteQ = 3  # fette per quadrante
    variante = 'inv'
else:
    anelli = 5
    fetteQ = 4  # fette per quadrante
    variante = 'inv'

fette = fetteQ * 4                  #calcolo fette per anello
n_quadranti = anelli * fette        #calcolo numero di settori totali

lista_immagini = os.listdir('Dataset/utkface')
size_dataset = len(lista_immagini)
print("Il dataset  ha una dimensione di " + str(size_dataset) + " elementi")
lista = []                  #lista in cui andranno inseriti tutti gli array che rappresentano la ragnatela del volto presente nelle immagini (una per ogni immagine)
num_volto = 0               #conta quante immagini abbiamo processato

for img in lista_immagini:
    if img.find(".jpg") > 0:
        path_immagine = "Dataset/utkface/" + str(img)
        foto = cv2.imread(path_immagine)
        volto = [0 for i in range(n_quadranti)]                   #array che contengono un contatore per ogni settore della ragnatela
        nomeimmagine = str(img)
        try:
            eta, sesso, razza, data = nomeimmagine.split('_')     #si ricavano età sesso e razza dal nome dell'immagine
        except ValueError:                                        #alcuni file non sono etichettati correttamente, verranno scartati
            continue

        rects, foto = responsive_resize(foto)                     #si individua il volto con la funzione detector presente all'interno di reponsive_resize()
        gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
        xnose, ynose = 0, 0     #coordinate naso
        raggio = 0              #raggio della ragnatela
        xlont, ylont = 0, 0     #coordinate del landmark più lontano
        distanza_punto = 0      #variabile che utilizziamo per calcolare la distanza dei landmark dal naso e trovare il raggio
        m = 0                   #coefficiente che diamo alla funzione aggiungi
        d = 0                   #distanza tra punto e naso che diamo alla funzione aggiungi
        #il ciclo viene utilizzato nel caso in cui in un'immagine abbiamo più di un volto, se non trova volti va all'immagine successiva
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            #stampa_volto(x, y, w, h, shape, foto)
            xnose = shape[33][0]        #il naso ha il landmark numero 33
            ynose = shape[33][1]        #il naso ha il landmark numero 33
            for (x, y) in shape:        #scegliamo il raggio guardando le coordinate più distanti
                distanza_punto = distanza(xnose, ynose, x, y)
                if(distanza_punto > raggio):
                    raggio = distanza_punto
                    xlont = x   #coordinata x del punto più lontano dal naso
                    ylont = y   #coordinata y del punto più lontano dal naso

            for(x,y) in shape:
                settore = [0,0,0]          #settore[0] = cerchio - settore[1] = quadrante - settore[2] = fetta
                if(y == ynose):            #calcola il coefficiente per ogni punto da utilizzare nella funzione aggiungi
                    m = 0
                else:
                    m = (x - xnose)/(y-ynose)
                m = abs(m)                 #valore assoluto di m
                d = distanza(xnose, ynose, x,y)
                aggiungi(xnose, ynose, raggio, x, y, d, m, anelli, numero_scelta-1)
            #aggiunto anelli e per settare i parametri della ragnatela in modo "responsive"
            #aggiungiamo all'array di settori le 3 informazioni derivate dal nome: eta, razza, sesso, nome file
            volto.append(int(eta))
            volto.append(int(razza))
            volto.append(int(sesso))
            volto.append(nomeimmagine)
            lista.append(volto)              #salviamo l'array dei settori nella lista
            num_volto = num_volto + 1        #contatore numero di immagini processate
        if (num_volto % 200) == 0:           #stampa l'avanzamento del processo
            print("Ho processato " + str(num_volto) + " elementi")

#viene salva la lista di ragnatele con le rispettive features all'interno di un file csv con il numero che indica la configurazione
write_list_to_file(lista, 'Dataset-'+str(anelli)+'C_'+str(fetteQ)+'S_'+str(variante)+'.csv')
