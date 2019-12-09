#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:30:06 2019

@author: raphaelletoubiana
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
from numpy import unravel_index
from PIL import Image
import time
from RWProba import label
from RWProba import randomWalker

def dataBase(file, size):
    dataSet=[]
    f=open(file,"r")
    while True:
        line=f.readline()
        if not line:
            break
        liste=list(map(float,line.split(',')))
        array=np.asarray(liste)
        matrix=np.reshape(array,(size,size))
        dataSet.append(matrix)
    f.close()
    return(dataSet)

dataSet7=dataBase("train.7.txt",16)
dataSet1=dataBase("train.1.txt",16)
 
#label des 5 premiers 1 et des 5 premiers 7
labell10=[label(4,8,1), label(1,1,-1), label(15,15,0) ]
labell11=[label(6,8,1), label(1,1,-1), label(15,15,0)]
labell12=[label(7,8,1), label(1,1,-1), label(15,15,0)]
labell13=[label(13,7,1), label(1,1,-1), label(15,15,0)]
labell14=[label(2,7,1), label(1,1,-1), label(15,15,0)]
labell70=[label(9,8,1), label(5,9,-1), label(15,15,0)]
labell71=[label(3,7,1), label(0,0,-1), label(15,15,0)]
labell72=[label(2,7,1), label(0,0,-1), label(15,15,0)]
labell73=[label(1,6,1), label(0,0,-1), label(15,15,0)]
labell74=[label(1,8,1), label(0,0,-1), label(15,15,0)]

#tableau des labels pour les images de 1 et de 7
labell7=[labell70,labell71,labell72,labell73,labell74]
labell1=[labell10,labell11,labell12,labell13,labell14]
print(labell11)
#mat=randomWalker(dataSet1[1],labell11[1])

def result(dataSet,label):
    results=[]
    temps=[]
    for i in range (5):
        start_time = time.time()
        mat=randomWalker(dataSet[i],label[i])
        results.append(mat)
        print(temps.append(time.time() - start_time))
        return (results)
    
def erreur(dataSet,result):
    erreur=[]
    for i in range (5):
        e=0
        for j in range (16):
            for k in range (16):
                if result[i][j][k]==0:
                    result[i][j][k]==-1
                e=e+(result[i][j][k]-dataSet[i][j][k])**2
        erreur.append(e)
    return(erreur)

#result1=result(dataSet1,labell1)

#im = Image.fromarray(np.uint8(dataSet[22] * 255) , 'L')
#im.save('922.jpg')
#im = Image.fromarray(np.uint8(dataSet[23] * 255) , 'L')
#im.save('923.jpg')
#im = Image.fromarray(np.uint8(dataSet[24] * 255) , 'L')
#im.save('924.jpg')
#im = Image.fromarray(np.uint8(dataSet[25] * 255) , 'L')
#im.save('925.jpg')
#im = Image.fromarray(np.uint8(dataSet[26] * 255) , 'L')
#im.save('926.jpg')
#im = Image.fromarray(np.uint8(dataSet[27] * 255) , 'L')
#im.save('927.jpg')
#im = Image.fromarray(np.uint8(dataSet[28] * 255) , 'L')
#im.save('928.jpg')
#im = Image.fromarray(np.uint8(dataSet[29] * 255) , 'L')
#im.save('929.jpg')
#im = Image.fromarray(np.uint8(dataSet[30] * 255) , 'L')
#im.save('930.jpg')

#im = Image.fromarray(np.uint8(dataSet[21] * 255) , 'L')
#im.save('921.jpg')
#im = Image.fromarray(np.uint8(dataSet[20] * 255) , 'L')
#im.save('920.jpg')
#im = Image.fromarray(np.uint8(dataSet[19] * 255) , 'L')
#im.save('919.jpg')
#im = Image.fromarray(np.uint8(dataSet[18] * 255) , 'L')
#im.save('918.jpg')
#im = Image.fromarray(np.uint8(dataSet[17] * 255) , 'L')
#im.save('917.jpg')
#im = Image.fromarray(np.uint8(dataSet[16] * 255) , 'L')
#im.save('916.jpg')
#im = Image.fromarray(np.uint8(dataSet[15] * 255) , 'L')
#im.save('915.jpg')
#im = Image.fromarray(np.uint8(dataSet[14] * 255) , 'L')
#im.save('914.jpg')
#im = Image.fromarray(np.uint8(dataSet[13] * 255) , 'L')
#im.save('913.jpg')

#im = Image.fromarray(np.uint8(dataSet[12] * 255) , 'L')
#im.save('912.jpg')
#im = Image.fromarray(np.uint8(dataSet[11] * 255) , 'L')
#im.save('911.jpg')
#im = Image.fromarray(np.uint8(dataSet[10] * 255) , 'L')
#im.save('910.jpg')
#im = Image.fromarray(np.uint8(dataSet[9] * 255) , 'L')
#im.save('99.jpg')
#im = Image.fromarray(np.uint8(dataSet[8] * 255) , 'L')
#im.save('98.jpg')
#im = Image.fromarray(np.uint8(dataSet[7] * 255) , 'L')
#im.save('97.jpg')
#im = Image.fromarray(np.uint8(dataSet[6] * 255) , 'L')
#im.save('96.jpg')
#im = Image.fromarray(np.uint8(dataSet[5] * 255) , 'L')
#im.save('95.jpg')
#im = Image.fromarray(np.uint8(dataSet[4] * 255) , 'L')
#im.save('94.jpg')
#im = Image.fromarray(np.uint8(dataSet[3] * 255) , 'L')
#im.save('93.jpg')
#im = Image.fromarray(np.uint8(dataSet[2] * 255) , 'L')
#im.save('92.jpg')
#im = Image.fromarray(np.uint8(dataSet[1] * 255) , 'L')
#im.save('91.jpg')
#im = Image.fromarray(np.uint8(dataSet[0] * 255) , 'L')
#im.save('90.jpg')