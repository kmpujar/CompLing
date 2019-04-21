from __future__ import division
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy
import operator
import keras
import tsne
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
import matplotlib.pylab as pylab
# import matplotlib as plt


def getFeaturenTagVector(file, trainTagVector=None, trainFeat=None, trFeatIndex=None):
	wordDict={}
	wordDict1={}
	tagDict={}
	features=[]
	with open(file) as fin:
		lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
	for nr,rows in enumerate(lines):
		word=str(rows[2])
		tagDim=str(rows[5])
		if tagDim=='_':
			continue
		check=(nr,word)
		if check not in wordDict1 and check not in wordDict:
			wordDict1[check]=list()
			wordDict[check]=len(wordDict)
			wordLen=len(word)
			if trFeatIndex==None:
				if(wordLen>0):
					features.append("PRE"+(word[0:1]))
					features.append("SUF"+(word[wordLen-1]))
				if(wordLen>1):
					features.append("PRE"+(word[0:2]))
					features.append("SUF"+(word[wordLen-2:]))
				if(wordLen>2):
					features.append("PRE"+(word[0:3]))
					features.append("SUF"+(word[wordLen-3:]))

			tags=tagDim.split(';')
			for t in tags:
				if trainTagVector!=None and t in trainTagVector:
					wordDict1[check].append(t)
					if t not in tagDict:
						tagDict[t]=trainTagVector[t]
				elif trainTagVector==None:
					wordDict1[check].append(t)
					if t not in tagDict:
						tagDict[t]=len(tagDict)


	if trFeatIndex==None:
		counter=Counter(features)
		freqFeat=dict(counter.most_common(901))

		featIndex = {w:i for i, w in enumerate(freqFeat)}
	else:
		featIndex = trFeatIndex
	for fi,ff in enumerate(featIndex):
		if fi<5:
			print(ff)
			print(featIndex[ff])

	trainVector=numpy.zeros((len(wordDict), 901))


	for f in featIndex:
		fInd=featIndex[f]
		for di in wordDict:
			d=di[1]
			wInd=wordDict[di]
			trainVector[wInd][900]=1
			wordLen=len(str(d))
			if len(f)==4 and wordLen>0:
				if(f[0:3]=='PRE' and f[3:]==d[0]):
					trainVector[wInd][fInd]=1
				if(f[0:3]=='SUF' and f[3:]==d[wordLen-1]):
					trainVector[wInd][fInd]=1
			elif len(f)==5 and wordLen>1:
				if(f[0:3]=='PRE' and f[3:]==d[0:2]):
					trainVector[wInd][fInd]=1
				elif(f[0:3]=='SUF' and f[3:]==d[wordLen-2]):
					trainVector[wInd][fInd]=1
			elif len(f)==6 and wordLen>2:
				if(f[0:3]=='PRE' and f[3:]==d[0:3]):
					trainVector[wInd][fInd]=1
				elif(f[0:3]=='SUF' and f[3:]==d[wordLen-3]):
					trainVector[wInd][fInd]=1

	tagVector=numpy.zeros((len(wordDict1), len(tagDict)))

	for w in wordDict1:
		i=wordDict[w]
		for t in tagDict:
			if(t in wordDict1[w]):
				k=tagDict[t]
				tagVector[i][k]=1
	return((wordDict,wordDict1,tagDict,features,trainVector,tagVector,featIndex))

def kerasMaxEnt(Xmatrix,Ymatrix,x_test):
	retTag=pd.DataFrame()
	model = keras.models.Sequential()
	model.add(Dense(Ymatrix.shape[1], input_shape=(901,),activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	model.fit(Xmatrix, Ymatrix,  batch_size=32,epochs=5,verbose=1)
	predictions = model.predict(x_test)
	discretePreds = predictions > .5
	return discretePreds

def trainData(trainVector,tagVector,taglen,testWordVector, testTagVector):
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	initW = numpy.random.randn(901, taglen)
	W = tf.Variable(initW, dtype=tf.float64)
	batch_size=32
	eta = .01
	loss_batch=[]
	score = tf.matmul(x, W)
	prob = 1 / (1 + tf.exp(-score))
	prob = tf.clip_by_value(prob, 0.000001, 0.999999)
	logPY = y * tf.log(prob) + (1-y) * tf.log(1 - prob)
	meanLL = -tf.reduce_mean(logPY)
	train_op = tf.train.AdamOptimizer(0.005).minimize(meanLL)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	wValue = sess.run(W)

	for i in range(5):
		for index, offset in enumerate(range(0, tagVector.shape[0], batch_size)):
			remaining = tagVector.shape[0] - offset
			if remaining < batch_size:
				x_batch, y_batch = trainVector[offset : offset + remaining], tagVector[offset : offset + remaining]
			else:
				x_batch, y_batch = trainVector[offset : offset + batch_size], tagVector[offset : offset + batch_size]
			loss, d = sess.run([meanLL,train_op], {x : x_batch, y : y_batch})
			print(loss)
			loss_batch.append(loss)
			wValue = sess.run(W)
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	score = tf.matmul(x, wValue)
	prob = 1 / (1 + tf.exp(-score))
	dev_sess = tf.Session()
	predictedProb = dev_sess.run(prob, {x : testWordVector, y : testTagVector})
	return(predictedProb)

def trainKerasMultiLayer(Xmatrix,Ymatrix,x_test):
	layers=3
	batchNorm=True
	lastHiddenLayer=None
	model = keras.models.Sequential()
	model.add(Dense(128, input_shape=(901,),activation="sigmoid"))
	model.add(keras.layers.normalization.BatchNormalization())
	model.add(Dense(128, input_shape=(901,),activation="sigmoid"))
	model.add(Dense(Ymatrix.shape[1], input_shape=(901,),activation="sigmoid"))
	for ii in range(0,layers):
		model.add(Dense(128, input_shape=(901,),activation="relu",name=str(ii)))
		if(ii<layers and batchNorm):
			model.add(keras.layers.normalization.BatchNormalization())
	model.add(Dense(Ymatrix.shape[1], input_shape=(901,),activation="sigmoid"))#, input_shape=(901,),activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	lastHiddenLayer=keras.models.Model(inputs=model.input,outputs=model.get_layer(str(layers-1)).output)
	model.fit(Xmatrix, Ymatrix,  batch_size=32,epochs=1,verbose=1)
	predictions = model.predict(x_test)
	discretePreds = predictions > .5
	return((discretePreds,lastHiddenLayer))

def main():
	trainFile='../Data/UD_English-EWT/en_ewt-um-train.conllu'
	testFile='../Data/UD_English-EWT/en_ewt-um-dev.conllu'
	trainWordDict,trainWordDictList,trainTagDict,trainFeat,trainWordVector,trainTagVector,featIndex=getFeaturenTagVector(trainFile)
	print(trainWordVector.shape)
	print(trainTagVector.shape)
	testWordDict,testWordDictList,testTagDict,testFeat,testWordVector,testTagVector,featIndex=getFeaturenTagVector(testFile,trainTagDict,trainFeat,featIndex)
	print(testWordVector.shape)
	print(testTagVector.shape)
	print(len(testTagVector[0]))
	# predictedProb=trainData(trainWordVector,trainTagVector,len(trainTagDict),testWordVector,testTagVector)
	predictedProb,LHL=trainKerasMultiLayer(trainWordVector,trainTagVector,testWordVector)
	print(predictedProb.shape)
	proposed_tag = defaultdict(list)
	correct=0
	for word in testWordDict:
		for tag in testTagDict:
			wordInd = testWordDict[word]
			tagInd = testTagDict[tag]
			if predictedProb[wordInd][tagInd] > 0.5:
				proposed_tag[word].append(tag)
		actSet=set(testWordDictList[word])
		proposedSet=set(proposed_tag[word])
		if actSet==proposedSet:
			print("True1 Word:	",word[1])
			print("True Actual:	",testWordDictList[word])
			print("True Predicted:	",proposed_tag[word])
			correct+=1#len(actSet.intersection(proposedSet))
			print()
	print(correct)
	print(correct/len(testWordDictList))

	sample = testWordVector[:200]
	intermediateOP = LHL.predict(sample)
	reducedDim = tsne.tsne(intermediateOP)

	pylab.scatter(reducedDim[:, 0], reducedDim[:, 1], 20)
	pylab.title("English")
	# pylab.xlabel("reducedDim[:, 0]")
	# pylab.ylabel("reducedDim[:, 1]")
	"""Add labels for each point"""
	wordL=list(testWordDictList.keys())
	for inx in range(200):
		word=wordL[inx][1]
		pylab.annotate(word, (reducedDim[inx, 0], reducedDim[inx, 1]))
	pylab.show()

	prop=0
	for word in testWordDictList:
		templist=testWordDictList[word]
		proposedTagList=proposed_tag[word]
		for t in templist:
			if(t in proposedTagList):
				prop+=1
	print(prop)


if __name__ == "__main__":
	main()
