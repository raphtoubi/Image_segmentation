import numpy as np
import matplotlib.pyplot as plt
import scipy 
from numpy import unravel_index
from PIL import Image
import time



"""TO DO : parallelize the work"""
"""TO DO : plot how much the error and execution time change with DistanceWeight : Find the optimal value for Beta"""
"""TO DO : récupérer toute la bdd en numpy"""
"""TO DO : make the edge cases a bit better by sending N diffusions and applying it for everycase where there is an equlity..."""


###Global variables
LengthPath = 1000  #How many steps are to be done when we start a path from a seed
NumberOfPathsStarted = 1000 #How many paths we start from each seed-type at the start of the algorithm
DistanceWeight = 5 #How much we penalize a difference in color

###Class representing a label
class label:
    def __init__(self, line, column, label):
        self.line = line
        self.column = column
        self.label = label


###Functions to chose the following cell

def goToNextPixelRandomly(currentPixelLine, currentPixelColumn, intensityMatrix, numberOfPassesMatrix, lengthMatrix, heightMatrix):
    newPixel = chooseNextPixelRandomly(currentPixelLine, currentPixelColumn, intensityMatrix, lengthMatrix, heightMatrix)     
    numberOfPassesMatrix[newPixel[0], newPixel[1]] = numberOfPassesMatrix[newPixel[0], newPixel[1]]+1   #we have now visited this new cell once
    return [newPixel[0], newPixel[1]]  #we return the new cell we should go to
    
    
def chooseNextPixelRandomly(currentPixelLine, currentPixelColumn, intensityMatrix, lengthMatrix, heightMatrix):
    currentIntensity = intensityMatrix[currentPixelLine, currentPixelColumn]
    distanceFromSurroundings = np.array([0.,0.,0.,0.]) #We will have pixels in this order : top, right, bottom, left
    if(currentPixelLine!=0):
        distanceFromSurroundings[0] = np.exp(-DistanceWeight*((currentIntensity-intensityMatrix[currentPixelLine-1, currentPixelColumn])**2))
    if(currentPixelColumn!=lengthMatrix-1):
        distanceFromSurroundings[1] = np.exp(-DistanceWeight*((currentIntensity-intensityMatrix[currentPixelLine, currentPixelColumn+1])**2))
    if(currentPixelLine!=heightMatrix-1):
        distanceFromSurroundings[2] = np.exp(-DistanceWeight*((currentIntensity-intensityMatrix[currentPixelLine+1, currentPixelColumn])**2))
    if(currentPixelColumn!=0):
        distanceFromSurroundings[3] = np.exp(-DistanceWeight*((currentIntensity-intensityMatrix[currentPixelLine, currentPixelColumn-1])**2))
    distanceFromSurroundings = distanceFromSurroundings / np.sum(distanceFromSurroundings)
    return randomDraw(currentPixelLine, currentPixelColumn, distanceFromSurroundings)
    
    
def randomDraw(currentPixelLine, currentPixelColumn, distanceFromSurroundings):
    randomValue = np.random.rand()
    distanceCumSum = np.cumsum(distanceFromSurroundings) #Will make it a bit easier for the rest of this function
    if(randomValue<=distanceCumSum[0]):
        return np.array([currentPixelLine-1, currentPixelColumn])
    elif(randomValue<=distanceCumSum[1]):
        return np.array([currentPixelLine, currentPixelColumn+1])
    elif(randomValue<=distanceCumSum[2]):
        return np.array([currentPixelLine+1, currentPixelColumn])
    else:
        return np.array([currentPixelLine, currentPixelColumn-1])
    #We just chose randomly in which cell we go depending on the probilities given the difference in intensities
        
###Implementation of the algorithm

def randomWalker(intensityMatrix, seedLabels): #intensityMatrix is a np.array while seedLabels is a np.array of labels
    numberOfPathsToStartForThisLabel = initializeListLabels(seedLabels)
    shape = np.shape(intensityMatrix)
    numberOfPassesDictionaryForEachSeed = initializeNumberOfPassesDictionaryForEachSeed(seedLabels, shape)
    for seed in seedLabels:
        for pathNumber in range((int)(numberOfPathsToStartForThisLabel[seed.label])):
            startDiffusionFromOneSeed(seed, numberOfPassesDictionaryForEachSeed[seed.label], intensityMatrix, shape[1], shape[0])
    labelMatrix = numberOfPassesToLabels(numberOfPassesDictionaryForEachSeed, seedLabels, shape) #list of the labels
    postAlgorithmTreatment(intensityMatrix, labelMatrix, shape[1], shape[0]) #we treat the cells that have not been reached
    return labelMatrix
    
            
def startDiffusionFromOneSeed(seed, numberOfPassesMatrix, intensityMatrix, lengthMatrix, heightMatrix):
    currentPixel = [seed.line, seed.column]
    for i in range(LengthPath):
        currentPixel = goToNextPixelRandomly(currentPixel[0], currentPixel[1], intensityMatrix, numberOfPassesMatrix, lengthMatrix, heightMatrix)
    
def numberOfPassesToLabels(numberOfPassesDictionaryForEachSeed, seedLabels, shape):
    labelMatrix = np.zeros(shape)
    for seed in seedLabels:
        labelMatrix[seed.line][seed.column] = seed.label
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(labelMatrix[i][j]==0):
                labelWithMaxPassage = -1
                maxPassage = -1
                for seedLabel in numberOfPassesDictionaryForEachSeed:
                    if(numberOfPassesDictionaryForEachSeed[seedLabel][i][j]>maxPassage):
                        maxPassage = numberOfPassesDictionaryForEachSeed[seedLabel][i][j]
                        labelWithMaxPassage = seedLabel
                if((maxPassage!=0) and (maxPassage!=-1)):
                    labelMatrix[i][j] = labelWithMaxPassage
                else:
                    labelMatrix[i][j] = -np.Infinity
    return labelMatrix
    
###Initiation of seeding lists
    
def initializeListLabels(seedLabels):
    listLabelsWithNumberOfSeeds = []  #Will contain a list of [label.number, number of seeds with this label]
    maxLabel = -1
    for label in seedLabels:
        updateListLabelsWithOneLabel(listLabelsWithNumberOfSeeds, label)  #we update the number of seeds of each type with this seed
        maxLabel = max(maxLabel, label.label)
    numberOfPathsToStartForThisLabel = np.zeros(maxLabel+1) #We will just keep a list for which the element i tells the number of paths to start for a seed of label i
    for label in listLabelsWithNumberOfSeeds:
        numberOfPathsToStartForThisLabel[label[0]] = (NumberOfPathsStarted/label[1])
    return np.array(numberOfPathsToStartForThisLabel)

def updateListLabelsWithOneLabel(listLabels, label):
    for index in range(len(listLabels)): #if there is another seed of this type, we just add that there is a new one
        if(listLabels[index][0]==label.label):
            listLabels[index][1]=listLabels[index][1]+1
            return
    listLabels.append([label.label, 1])  #else, we add this type of label
    
def initializeNumberOfPassesDictionaryForEachSeed(seedLabels, shape):
    numberOfPassesDictionaryForEachSeed = {}
    for seed in seedLabels:
        numberOfPassesDictionaryForEachSeed[seed.label] = np.zeros(shape)
    return numberOfPassesDictionaryForEachSeed
    
    
###Diffusion from a cell if its type is unknown after the diffusion
def postAlgorithmTreatment(intensityMatrix, labelMatrix, lengthMatrix, heightMatrix):
    #find the cells where no diffusion has gone through and put a label on them through ONE new diffusion starting from this cell
    for i in range(heightMatrix):
        for j in range(lengthMatrix):
            if(labelMatrix[i][j]== -np.Infinity):
                oneDiffusionFromUnknownCell(labelMatrix, intensityMatrix, [i, j], lengthMatrix, heightMatrix)
                
def oneDiffusionFromUnknownCell(labelMatrix, intensityMatrix, unknownCell, lengthMatrix, heightMatrix):
    currentPixel = [unknownCell[0], unknownCell[1]]
    while(labelMatrix[currentPixel[0]][currentPixel[1]]== -np.Infinity): #We keep on going to the next Cell until we are on a cell where the color has been determined
        currentPixel = chooseNextPixelRandomly(currentPixel[0], currentPixel[1], intensityMatrix, lengthMatrix, heightMatrix)
    labelMatrix[unknownCell[0], unknownCell[1]]=labelMatrix[currentPixel[0], currentPixel[1]]


### Database


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

 
dataSet6=dataBase("train.6.txt",16)
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

#Fonction qui renvoie une matrice d'array après passage dans le RW
def result(dataSet,label):
    results=[]
    for i in range (5):
        start_time = time.time()
        mat=randomWalker(dataSet[i],label[i])
        results.append(mat)
        print(time.time() - start_time)
    return (results)


#Fonction qui renvoie l'erreur pour chaque image
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
        print(e)
    return(erreur)

seedWhite = label(13,6,1)
seedBlack = label(0,0,-1)
seedBlack3=label(15,15,0)
labell=[seedWhite,seedBlack,seedBlack3]


#erreur=[2759.40918899999,2286.547189,1815.76718899999,1439.005189,1263.209189,1121.737189,1148.733189,1090.545189,1133.557189,1063.775189,1075.003189,1047.011759,1059.047189,1043.826944,1055.646944,1042.3,1034.3,1078.097189,1073.896944,1043.456944]
#temps=[22.991,24.8955468177795,30.1516654491424,21.1445846557617,21.7229957962036,22.0986188030243,20.6178803253173,20.5113147449493,21.8379088211059,23.1846664237976,20.6865283885955,21.8398869285583,21.3428188419342,30.3113016700744,28.626331167221,37.2656201362609,45.3336567020415,34.5216775035858,30.343772239685,75.5977513980865]
#beta=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#plt.plot(beta,erreur)
#plt.title("Impact du paramètre beta sur l'erreur")
#plt.xlabel("Beta")
#plt.ylabel("Erreur")
#plt.show()

#plt.plot(beta,temps)
#plt.title("Impact du paramètre beta sur le temps d'exécution")
#plt.xlabel("Beta")
#plt.ylabel("Temps d'exécution")
#plt.show()






