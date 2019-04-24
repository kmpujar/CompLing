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
import tsne
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib.pylab as pylab
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
from kmeans import Kmeans


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
			trainingSetTripletTags[i]=(triple,x,y)
			i+=1
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
	actual=defaultdict(int)
	with open(file, encoding='utf-8') as fin:
		for line in fin:
			if line[0]!='#' and line[0]!=' ':
				if line!='\n':
					lines.append(line.split('\t'))
				else:
					lines.append(['\n'])
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
			actual[t]+=1

		left=current
		current=right

	wordLength=np.array(wordLength)
	maxwordLength=int(np.percentile(wordLength,95))
	return ((wordDict,wordTriplet,maxwordLength,tripletTagRecord,tagDict,tagDim,actual))



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


def printEvalMetrics(actual,matched,proposed,tags,fileName):
	f1 = lambda P,R : 2*(P*R)/(P+R)
	matchedCount=sum(matched.values())
	proposedCount=sum(proposed.values())
	actualCount=sum(actual.values())
	precision=matchedCount/proposedCount
	recall=matchedCount/actualCount
	print("The number of non-zero tags is: " + str(len(matched)))
	f=open(fileName,"a+")
	f.write("The number of non-zero tags is: " + str(len(matched))+"\n")
	f.write("Over all precision:  "+str(precision)+"\n")
	f.write("Over all recall:  "+str(recall)+"\n")
	f.write("Overall F1-score:  "+str(f1(precision,recall))+"\n")
	f.write("\n")
	for feat in tags.values():
		tagDim=feat.split(';')
		for t in tagDim:
			if feat not in matched:
				P = 1.0
				R = 0.0
			else:
				P = matched[feat] / proposed[feat]
				R = matched[feat] / actual[feat]
			f.write("Tag  "+t+"\n")
			f.write("Precision:  " + str(P)+"\n")
			f.write("Recall:  " + str(R)+"\n")
			f.write("F1-score:  " + str(f1(P, R))+"\n")
			f.write("___________________________________"+"\n")
	f.write("\n")
	f.close()

def tsnePlot(inputMatrix,auxModel,trainingSetTripletTags,language):
	size=50
	if "Sanskrit" in language:
		size=len(trainingSetTripletTags)
	sample = inputMatrix[:size]
	print(len(trainingSetTripletTags))
	samplePred = auxModel.predict(sample)
	tsneRedux = tsne.tsne(samplePred)
	pylab.figure()
	pylab.scatter(tsneRedux[:, 0], tsneRedux[:, 1], 20)
	pylab.title(language)
	pylab.xlabel("tsneRedux[:, 0]")
	pylab.ylabel("tsneRedux[:, 1]")
	for inx in range(size):
		c,x,y=trainingSetTripletTags[inx]
		pylab.annotate(inx,(tsneRedux[inx, 0], tsneRedux[inx, 1]))
	pylab.savefig(language+".png")


def main():
	trainFiles=['./Data/en_ewt-um-train.conllu','./Data/es_ancora-um-train.conllu','./Data/ru_gsd-um-train.conllu','./Data/tr_imst-um-train.conllu','./Data/zh_gsd-um-train.conllu','./Data/sa_ufal-um-train.conllu']
	testFiles=['./Data/en_ewt-um-dev.conllu','./Data/es_ancora-um-dev.conllu','./Data/ru_gsd-um-dev.conllu','./Data/tr_imst-um-dev.conllu','./Data/zh_gsd-um-dev.conllu','./Data/sa_ufal-um-dev.conllu']
	lang=['English', 'Spanish', 'Russian', 'Turkish', 'Chinese', 'Sanskrit']
	numClustersList=[8,16,32]
	# trainFiles=['./Data/en_ewt-um-train.conllu','./Data/es_ancora-um-train.conllu','./Data/ru_gsd-um-train.conllu','./Data/tr_imst-um-train.conllu','./Data/zh_gsd-um-train.conllu','./Data/sa_ufal-um-train.conllu']
	# testFiles=['./Data/en_ewt-um-dev.conllu','./Data/es_ancora-um-dev.conllu','./Data/ru_gsd-um-dev.conllu','./Data/tr_imst-um-dev.conllu','./Data/zh_gsd-um-dev.conllu','./Data/sa_ufal-um-dev.conllu']
	# lang=['English', 'Spanish', 'Russian', 'Turkish', 'Chinese', 'Sanskrit']
	# numClustersList=[32]
	for li in range(len(lang)):
		for numClusters in numClustersList:
			word_Dict,triplets,maxLen,tripletTagRecord,tagDict,tagDim,actual=getTriplet(trainFiles[li])
			charIndDict,indCharDict=getCharDict(word_Dict.keys())
			trainingData,trainingSetTripletTags=getTrainingSet(triplets)
			charsetlen=len(set(charIndDict))
			inputMatrix = np.zeros((len(trainingData), maxLen, charsetlen))
			outputMatrix = np.zeros((len(trainingData), maxLen, charsetlen))

			llen = 0


			for dataCount, (x, y) in enumerate(trainingData):
				for cou,ch in enumerate(x):
					if llen < maxLen:
						inputMatrix[dataCount][llen][charIndDict[ch]] = 1
						llen += 1
				if len(x)< maxLen:
					for l in range(llen, maxLen):
						inputMatrix[dataCount][l][0] = 1

				llen=0
				for cou,ch in enumerate(y):
					if llen < maxLen:
						outputMatrix[dataCount][llen][charIndDict[ch]] = 1
						llen += 1
				if len(y) < maxLen:
					for l in range(llen, maxLen):
						outputMatrix[dataCount][l][0] = 1



			auxilliaryModel=runLSTM(inputMatrix,outputMatrix,charsetlen,maxLen,1,1)
			eightDim=auxilliaryModel.predict(inputMatrix)
			randDataPoints=set()
			# numClusters=32
			clusters = defaultdict(list)
			dataPointDict={}
			pointList=[]


			for id,p in enumerate(eightDim):
				dataPointDict[tuple(p)]=id
				pointList.append(p)

			while(len(randDataPoints)<numClusters):
				randDataPoints.add(np.random.randint(0, dataCount))

			for i,ii in enumerate(randDataPoints):
				clusters[i] = [[]]
				clusters[i][0]=eightDim[ii]

			print("Evaluating Kmeans for "+lang[li]+" with "+str(numClusters)+" clusters.")
			km=Kmeans(clusters,dataPointDict,pointList)
			numIters=10
			for i in range(numIters):
				print("Iteration "+str(i)+"...")
				for point in km.pointList:
					km.getCluster(point)
				try:
					km.getNewCentroid()
				except Exception as e:
					print(e)
				if i != numIters-1:
					km.clearPoints()

			kmObjVal=km.calcMean(dataCount)
			matchedTags, proposedTags=km.kmEval(trainingSetTripletTags,tripletTagRecord)
			f=open(lang[li]+"_matchedTags_"+str(numClusters),"w+")
			for m in matchedTags:
				f.write(m+"\n")

			# print("Mean distance to the cluster center is: " + str(kmObjVal))
			# printEvalMetrics(actual,matchedTags,proposedTags,tagDict,lang[li]+str(numClusters))
			# if numClusters==32:
			# 	tsnePlot(inputMatrix,auxilliaryModel,trainingSetTripletTags,lang[li]+str(numClusters))

if __name__ == "__main__":
	main()
