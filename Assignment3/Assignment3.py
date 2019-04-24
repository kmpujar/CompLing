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
		word=str(rows[1])
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

	if trainTagVector!=None:
		for t in trainTagVector:
			tagDict[t]=trainTagVector[t]

	if trFeatIndex==None:
		counter=Counter(features)
		freqFeat=dict(counter.most_common(1001))

		featIndex = {w:i for i, w in enumerate(freqFeat)}
	else:
		featIndex = trFeatIndex

	trainVector=numpy.zeros((len(wordDict), 1001))


	for f in featIndex:
		fInd=featIndex[f]
		for di in wordDict:
			d=di[1]
			wInd=wordDict[di]
			trainVector[wInd][1000]=1
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
	model.add(Dense(Ymatrix.shape[1], input_shape=(1001,),activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	model.fit(Xmatrix, Ymatrix,  batch_size=32,epochs=5,verbose=1)
	predictions = model.predict(x_test)
	discretePreds = predictions > .5
	return discretePreds

def trainData(trainVector,tagVector,taglen,testWordVector, testTagVector):
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	initW = numpy.random.randn(1001, taglen)
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

def trainKerasMultiLayer(Xmatrix,Ymatrix,x_test,layers=3):
	lastHiddenLayer=None
	batchNorm=True
	model = keras.models.Sequential()
	model.add(Dense(Ymatrix.shape[1], input_shape=(1001,),activation="sigmoid"))
	# model.add(keras.layers.normalization.BatchNormalization())
	# model.add(Dense(128, input_shape=(1001,),activation="sigmoid"))
	# model.add(Dense(Ymatrix.shape[1], input_shape=(1001,),activation="sigmoid"))
	for ii in range(0,layers):
		model.add(Dense(128, input_shape=(1001,),activation="relu",name=str(ii)))
		if(batchNorm):
			model.add(keras.layers.normalization.BatchNormalization())
	model.add(Dense(Ymatrix.shape[1],activation="sigmoid"))#, input_shape=(1001,),activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	if layers>1:
		lastHiddenLayer=keras.models.Model(inputs=model.input,outputs=model.get_layer(str(layers-2)).output)
	model.fit(Xmatrix, Ymatrix,  batch_size=32,epochs=5,verbose=2)
	predictions = model.predict(x_test)
	# discretePreds = predictions > .5
	return((predictions,lastHiddenLayer))

def printEvalMetrics(actual,matched,proposed,tags,fileName,bl=True):
	f1 = lambda P,R : 2*(P*R)/(P+R)
	matchedCount=sum(matched.values())
	proposedCount=sum(proposed.values())
	actualCount=sum(actual.values())
	precision=matchedCount/proposedCount
	recall=matchedCount/actualCount
	f=open(fileName,"a+")
	f.write("Over all precision:  "+str(precision)+"\n")
	f.write("Over all recall:  "+str(recall)+"\n")
	f.write("Overall F1-score:  "+str(f1(precision,recall))+"\n")
	f.write("\n")
	for feat in tags:
		if bl==False:
			tagDim=feat.split(';')
		else:
			tagDim=[feat]
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

def main():
	trainFiles=['./Data/en_ewt-um-train.conllu','./Data/es_ancora-um-train.conllu','./Data/ru_gsd-um-train.conllu','./Data/tr_imst-um-train.conllu','./Data/zh_gsd-um-train.conllu','./Data/sa_ufal-um-train.conllu']
	testFiles=['./Data/en_ewt-um-dev.conllu','./Data/es_ancora-um-dev.conllu','./Data/ru_gsd-um-dev.conllu','./Data/tr_imst-um-dev.conllu','./Data/zh_gsd-um-dev.conllu','./Data/sa_ufal-um-dev.conllu']
	lang=['English', 'Spanish', 'Russian', 'Turkish', 'Chinese', 'Sanskrit']
	# trainFiles=['./Data/sa_ufal-um-train.conllu']
	# testFiles=['./Data/sa_ufal-um-dev.conllu']
	# lang=['Sanskrit']
	for i in range(len(lang)):
		trainWordDict,trainWordDictList,trainTagDict,trainFeat,trainWordVector,trainTagVector,featIndex=getFeaturenTagVector(trainFiles[i])
		testWordDict,testWordDictList,testTagDict,testFeat,testWordVector,testTagVector,featIndex=getFeaturenTagVector(testFiles[i],trainTagDict,trainFeat,featIndex)
		# predictedProb=trainData(trainWordVector,trainTagVector,len(trainTagDict),testWordVector,testTagVector)
		predictedProb,LHL=trainKerasMultiLayer(trainWordVector,trainTagVector,testWordVector)
		propposedTag = defaultdict(list)
		proposed=defaultdict(int)
		actual=defaultdict(int)
		matched=defaultdict(int)
		misclassified=defaultdict(int)
		misRate=defaultdict(list)
		misRateVal=defaultdict(int)
		correct=0
		for word in testWordDict:
			for tag in testTagDict:
				wordInd = testWordDict[word]
				tagInd = testTagDict[tag]
				try:
					if predictedProb[wordInd][tagInd] > 0.5:
						propposedTag[word].append(tag)
				except:
					f.write(e)

		f=open(lang[i]+"_words","w+")

		for word in testWordDictList:
			tags=testWordDictList[word]
			for tag in tags:
				actual[tag]+=1
				if tag not in propposedTag[word]:
					misclassified[tag]+=1
					misRate[tag].append(word)
			for tag in propposedTag[word]:
				proposed[tag]+=1
				if tag in tags:
					matched[tag]+=1
			actSet=set(testWordDictList[word])
			proposedSet=set(propposedTag[word])
			# print(actSet)
			# print(proposedSet)
			if actSet==proposedSet:
				correct+=1
			f.write("Word:  "+word[1]+"\n")
			f.write("Actual: ")
			for w in testWordDictList[word]:
				f.write(w+";")
			f.write("\n")
			f.write("Predicted:  ")
			for w in propposedTag[word]:
				f.write(w+";")
			f.write("\n")
			f.write('___________________________________\n')
		f.close()


		f=open(lang[i],"w+")
		f.write("Total words:  "+str(len(testWordDictList))+'\n')
		f.write("Accuracy:  "+str(correct/len(testWordDictList))+'\n')
		printEvalMetrics(actual,matched,proposed,testTagDict,lang[i])
		f.close()

		size=200
		if lang[i]=="Sanskrit":
			size=len(testWordDictList)
		sample = testWordVector[:size]
		intermediateOP = LHL.predict(sample)
		reducedDim = tsne.tsne(intermediateOP)
		pylab.figure()
		pylab.scatter(reducedDim[:, 0], reducedDim[:, 1], 20)
		pylab.title(lang[i])
		# pylab.xlabel("reducedDim[:, 0]")
		# pylab.ylabel("reducedDim[:, 1]")
		"""Add labels for each point"""
		wordL=list(testWordDictList.keys())
		for inx in range(size):
			try:
				word=wordL[inx][1]
			except:
				inx-=1
			pylab.annotate(word, (reducedDim[inx, 0], reducedDim[inx, 1]))
		pylab.savefig(lang[i]+".png")



if __name__ == "__main__":
	main()
