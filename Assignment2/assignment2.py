from __future__ import division
from collections import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy
import heapq

def getNextBatch( data, labels,num, indices=None):
	'''
	Return a total of `num` random samples and labels.
	'''
	if indices is None:
		idx = numpy.arange(0 , data.shape[0])
		numpy.random.shuffle(idx)
		idx = idx[:num]
	else:
		idx = []
		for ii in range(num):
			idx.append(indices.pop(0))
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]

	return numpy.asarray(data_shuffle), numpy.asarray(labels_shuffle)

def getFeaturenTagVector(file, trainTagVector=None, trainFeat=None, trFeatIndex=None, extraFeat=False):
	wordDict={}
	wordDict1={}
	tagDict={}
	features=[]
	with open(file) as fin:
		lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
	for nr,rows in enumerate(lines):
		word=str(rows[1])
		tagDim=str(rows[5])
		check=(nr,word)
		if check not in wordDict1 and check not in wordDict:
			wordDict1[check]=list()
			wordDict[check]=len(wordDict)
			wordLen=len(word)
			if trFeatIndex==None:
				if(wordLen>0):
					features.append("PRE"+(word[0:1]))
					features.append("SUF"+(word[wordLen-1]))
					if(extraFeat):
						if('कः' in word):
							features.append("F1कः")
						if('तः' in word):
							features.append("F2तः")
						if('सः' in word):
							features.append("F2सः")
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
			if(extraFeat):
				if len(f)==3 and wordLen>0:
					if(f[0:2]=="F1" and f[2] in d):
						trainVector[wInd][fInd]=1
					if(f[0:2]=="F2" and f[2] in d):
						trainVector[wInd][fInd]=1
					if(f[0:2]=="F3" and f[2] in d):
						trainVector[wInd][fInd]=1
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

def trainData(trainVector,tagVector,taglen,testWordVector, testTagVector,batch_size,fileName):

	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	initW = numpy.random.randn(1001, taglen)
	W = tf.Variable(initW, dtype=tf.float64)
	eta=0
	niters=0
	if batch_size==1:
		eta =0.01
		niters=1
	elif batch_size==32:
		eta=0.005
		niters=4
	else:
		eta=0.001
		niters=20
		if 'Sanskrit' in fileName:
			niters=250
	loss_batch=[]
	score = tf.matmul(x, W)
	prob = 1 / (1 + tf.exp(-score))
	prob = tf.clip_by_value(prob, 0.000001, 0.999999)
	logPY = y * tf.log(prob) + (1-y) * tf.log(1 - prob)
	meanLL = -tf.reduce_mean(logPY)
	train_op = tf.train.AdamOptimizer(eta).minimize(meanLL)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	wValue = sess.run(W)
	f=open(fileName,"w+")
	f.write("Loss:"+'\n')
	for i in range(niters):
		for index, offset in enumerate(range(0, tagVector.shape[0], batch_size)):
			remaining = tagVector.shape[0] - offset
			if remaining < batch_size:
				x_batch, y_batch = trainVector[offset : offset + remaining], tagVector[offset : offset + remaining]
			else:
				x_batch, y_batch = trainVector[offset : offset + batch_size], tagVector[offset : offset + batch_size]
			loss, d = sess.run([meanLL,train_op], {x : x_batch, y : y_batch})
			if index%250==0:
				f.write(str(loss)+'\n')
			loss_batch.append(loss)
			wValue = sess.run(W)
	f.write('\n')
	f.write('\n')
	f.close()
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	score = tf.matmul(x, wValue)
	prob = 1 / (1 + tf.exp(-score))
	dev_sess = tf.Session()
	predictedProb = dev_sess.run(prob, {x : testWordVector, y : testTagVector})
	return(predictedProb)

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
	extraFeats=[False]

	# lang=['Sanskrit']
	for i in range(len(lang)):
		if lang[i]=='Sanskrit':
			extraFeats=[False,True]
		for bs in extraFeats:
			trainWordDict,trainWordDictList,trainTagDict,trainFeat,trainWordVector,trainTagVector,featIndex=getFeaturenTagVector(trainFiles[i],extraFeat=bs)
			testWordDict,testWordDictList,testTagDict,testFeat,testWordVector,testTagVector,featIndex=getFeaturenTagVector(testFiles[i],trainTagDict,trainFeat,featIndex,extraFeat=bs)
			predictedProb=trainData(trainWordVector,trainTagVector,len(trainTagDict),testWordVector,testTagVector,128,lang[i]+"_1001_"+str(bs))

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

			f=open(lang[i]+"_1001_"+str(bs)+"_words","w+")

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
				if actSet==proposedSet:
					f.write("Word:  "+word[1]+"\n")
					f.write("Actual: ")
					for w in testWordDictList[word]:
						f.write(w)
					f.write("\n")
					f.write("Predicted:  ")
					for w in propposedTag[word]:
						f.write(w)
					f.write("\n")
					correct+=1
					f.write('___________________________________\n')
			f.close()
			if lang[i]=="Sanskrit":
				f=open(lang[i]+"_misclassified","w+")
				misclassifiedD=sorted(misclassified, key=misclassified.get, reverse=True)
				for j,m in enumerate(misclassifiedD):
					f.write("10 most misclassified tags are:"+'\n')
					if j<10:
						f.write(m)
					misRateVal[m]=misclassified[m]/actual[m]
				misRateValD=sorted(misRateVal, key=misRateVal.get, reverse=True)
				for j,m in enumerate(misRateValD):
					if j>2:
						break
					f.write("Tag:  "+str(m)+'\n')
					f.write("Misclassified words:"+'\n')
					for wc,w in enumerate(misRate[m]):
						if wc<25:
							f.write(w[1]+'\n')
					f.write('___________________________________\n')
				f.close()

			f=open(lang[i]+"_1001_"+str(bs),"a+")
			f.write("Total words:  "+str(len(testWordDictList))+'\n')
			f.write("Accuracy:  "+str(correct/len(testWordDictList))+'\n')
			printEvalMetrics(actual,matched,proposed,testTagDict,lang[i]+"_1001_"+str(bs))
			f.close()

if __name__ == "__main__":
	main()
