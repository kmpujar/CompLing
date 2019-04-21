from __future__ import division
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import heapq
import operator
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib as plt
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
from kmeans import Kmeans
#print(device_lib.list_local_devices())


def getTrainingSet(wordTriplet):
    trainingSet = []
    i=0
    trainingSetTripletTags={}
    for triple in (wordTriplet):
        num = len(wordTriplet[triple].keys())
        if num > 1:
            repeat = True
            while repeat:
                rand1 = np.random.randint(0, num)
                rand2 = np.random.randint(0, num)
                repeat = rand1 == rand2
            x = ''
            y = ''
            for ind, key in enumerate(wordTriplet[triple]):
                if ind == rand1:
                    x = key
                if ind == rand2:
                    y = key
            trainingSet.append((x, y))
            # tempList=[]
            # tempList.append(triple)
            # tempList.append(x)
            # tempList.append(y)
            trainingSetTripletTags[i]=(triple,x,y)
            i+=1
            # print(i)
            # print(trainingSetTripletTags)
            # exit()
    return (trainingSet,trainingSetTripletTags)


def getCharDict(cdict, flag=False):
    charIndDict=defaultdict(int)
    indCharDict=defaultdict(str)
    charIndDict['PAD']=0
    indCharDict[0]='PAD'
    l=0
    for word in cdict:
        for i in word:
            if i not in charIndDict:
                l+=1
                charIndDict[i]=l
                indCharDict[l]=i
                if flag and len(charIndDict)==26:
                    break
    return((charIndDict,indCharDict))

def getTriplet(file):
    wordDict={}
    # wordDict['<s>']=len(wordDict    )
    # wordDict['PAD']=len(wordDict)
    tagDict={}
    charSet=set()
    wordLength=[]
    wordTriplet=defaultdict(dict)
    tripletTagRecord=defaultdict(set)
    startChar='<s>'
    left=startChar
    current=startChar
    right=None
    startFlag=True
    tagDim=defaultdict(int)
    lines=[]
    with open(file, encoding='utf-8') as fin:
        for line in fin:
            if line[0]!='#' and line[0]!=' ':
                if line!='\n':
                    lines.append(line.split('\t'))
                else:
                    lines.append(['\n'])
        #lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
    l=0
    for rnum, rows in enumerate(lines):
        triTags=[]
        if rows[0]!='\n':
            try:
                triTags.append(lines[rnum-2][5])
            except:
                triTags.append('_')
            try:
                triTags.append(lines[rnum-1][5])
            except:
                triTags.append('_')
            try:
                triTags.append(lines[rnum][5])
            except:
                triTags.append('_')

            word=str(rows[1])
            tag=str(rows[5])
            tg=tag.split(';')
            for t in tg:
                tagDim[t]+=1
            wordDict[word]=l
            l+=1
            tagDict[word]=tag
            wordLength.append(len(word))
        else:
            startFlag=True
            continue
        if startFlag:
            left=startChar
            current=startChar
            startFlag=False
        right=word
        if current not in wordTriplet[(left,right)]:
            wordTriplet[(left,right)][current]=1
        else:
            wordTriplet[(left,right)][current]+=1
        # tripletTagRecord[(left,current,right)]=[triTags[0],triTags[1],triTags[2]]

        td=triTags[1].split(';')
        for t in td:
            tripletTagRecord[(left,current,right)].add(t)

        left=current
        current=right

    wordLength=np.array(wordLength)
    maxwordLength=int(np.percentile(wordLength,95))
    return ((wordDict,wordTriplet,maxwordLength,tripletTagRecord,tagDict,tagDim))



def runLSTM(input,output,charsetLen,maxLen,numDenseLayers,addLSTM=1):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, activation='tanh', input_shape=(maxLen, charsetLen)))

    for i in range(numDenseLayers):
        model.add(keras.layers.Dense(8, activation='relu', name=str(i+1)))
        if i+1 == numDenseLayers:
            prevLayer = str(numDenseLayers)
            auxModel = keras.models.Model(inputs=model.input,outputs=model.get_layer(prevLayer).output)

    model.add(keras.layers.RepeatVector(maxLen))

    for i in range(addLSTM):
        model.add(keras.layers.LSTM(64, activation="relu", return_sequences=True))
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.TimeDistributed(keras.layers.Dense(charsetLen, activation="softmax")))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(input, output, epochs=10, batch_size=128,verbose=2)
    return(auxModel)

def compare(s, t):
    return sorted(s) == sorted(t)

def main():
    file='es_ancora-um-train.conllu.txt'
    # trainData=getTriplet(file)
    word_Dict,triplets,maxLen,tripletTagRecord,tagDict,tagDim=getTriplet(file)
    print(len(triplets))
    print(maxLen)
    print(len(tripletTagRecord))
    print(len(tagDim))
    # word_Dict=trainData[0]
    # print(len(word_Dict))
    # word_Dict['<s>']=len(word_Dict)+1
    # word_Dict['PAD']=len(word_Dict)+1
    # triplets=trainData[1]
    # maxLen=trainData[2]
    # tripletTagRecord=trainData[3]
    # tagDict=trainData[4]
    # tagDim=trainData[5]
    # for t in triplets:
    #     print(str(t)+" : "+str(triplets[t]))
    # exit()
    charIndDict,indCharDict=getCharDict(word_Dict.keys())
    # charIndDict=charData[0]
    # indCharDict=charData[1]
    trainingData,trainingSetTripletTags=getTrainingSet(triplets)
    # for t in trainingSetTripletTags:
    #     print(str(t)+" : "+str(trainingSetTripletTags[t]))
    # exit()
    # print(len(trainingData))
    # print(trainingData[0])
    # print(trainingData[1])
    # print(len(trainingSetTripletTags))
    # print(maxLen)
    # exit()
    charsetlen=len(set(charIndDict))
    print(charsetlen)
    #print(charsetlen)
    with open('your_file.txt', 'w') as f:
        for item in charIndDict:
            f.write(str(item)+" : "+str(charIndDict[item]))
    inputMatrix = np.zeros((len(trainingData), maxLen, charsetlen))
    outputMatrix = np.zeros((len(trainingData), maxLen, charsetlen))

    llen = 0


    for dataCount, (x, y) in enumerate(trainingData):
        # print(x)
        # print(y)
        for cou,ch in enumerate(x):
            if llen < maxLen:
                inputMatrix[dataCount][llen][charIndDict[ch]] = 1
                llen += 1
        if len(x)< maxLen:
            #print("hix"+str(llen))
            for l in range(llen, maxLen):
                inputMatrix[dataCount][l][0] = 1

        llen=0
        for cou,ch in enumerate(y):
            if llen < maxLen:
                outputMatrix[dataCount][llen][charIndDict[ch]] = 1
                llen += 1
        if len(y) < maxLen:
            #print("hiy"+str(llen))
            for l in range(llen, maxLen):
                outputMatrix[dataCount][l][0] = 1



    auxilliaryModel=runLSTM(inputMatrix,outputMatrix,charsetlen,maxLen,1,1)
    eightDim=auxilliaryModel.predict(inputMatrix)
    randDataPoints=set()
    numClusters=32
    clusters = defaultdict(list)
    dataPointDict={}
    pointList=[]


    for id,p in enumerate(eightDim):
        dataPointDict[tuple(p)]=id
        pointList.append(p)

    while(len(randDataPoints)<numClusters):
        randDataPoints.add(np.random.randint(0, dataCount))
    print("rand len: "+str(len(randDataPoints)))

    for i,ii in enumerate(randDataPoints):
        clusters[i] = [[]]
        clusters[i][0]=eightDim[ii]

    print("num clust    "+str(len(clusters)))
    km=Kmeans(clusters,dataPointDict,pointList)
    #print(len(km.clusters[0]))
    # print(km.numClusters)
    # print(len(km.pointDict))
    print("done")
    numIters=10
    for i in range(numIters):
        print("Iter"+str(i))
        # print('before'+str(len(km.clusters[0][1:])))
        for point in km.pointList:
            km.getCluster(point)
        try:
            km.getNewCentroid()
            # print('after'+str(len(km.clusters[0][1:])))
        except Exception as e:
            print(e)
            #print("hello there")
            # i-=1
            # continue
            # randDataPoints=set()
            # km.clusters = defaultdict(list)
            # while(len(randDataPoints)<numClusters):
            #     randDataPoints.add(np.random.randint(0, dataCount))
            #
            # for i,ii in enumerate(randDataPoints):
            #     km.clusters[i] = [[]]
            #     km.clusters[i][0]=eightDim[i]
            #continue
        # print("Iter"+str(i))
        if i != numIters-1:
            km.clearPoints()
    # print("before eval")
    # print(len(km.clusters[0][1:]))
    kmObjVal=km.calcMean(dataCount)
    # print(len(km.clusters)
    matchedTags, proposedTags=km.kmEval(trainingSetTripletTags,tripletTagRecord)

    """Calculate precision, recall and F1 for each individual tag dimension"""
    mCount = 0
    metric=defaultdict(tuple)
    for tag in tagDim.keys():
        f1=0.0
        try:
            if tag in matchedTags:
                precision = matchedTags[tag] / proposedTags[tag]
                recall = matchedTags[tag] / tagDim[tag]
                metric.append((precision,recall, (2*precision*recall/(precision+recall))))
        except:
            continue

    """Calculate precision, recall and F1 in general"""

    # for i in matchedTags:
    #     print(i)

    # print(len(proposedTags))
    totalMatched = sum(matchedTags.values())
    totalProposed= sum(proposedTags.values())
    totalTagDim = sum(tagDim.values())

    precisionAll = totalMatched / totalProposed
    recAll = totalMatched / totalTagDim
    f1All= 2*precisionAll*recAll/(precisionAll+recAll)
    print("The number of non-zero tags is: " + str(len(matchedTags)))
    print("Mean distance to the cluster center is: " + str(kmObjVal))
    print("Precision: " + str(precisionAll))
    print("Recall: " + str(recAll))
    print("F1-score: " + str(f1All))

if __name__ == "__main__":
    main()
