from __future__ import division
from collections import defaultdict
from collections import Counter
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


# fileList=['./Data/UD_Spanish-AnCora/es_ancora-um-train.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-train.conllu','./Data/UD_Russian-GSD/ru_gsd-um-train.conllu','./Data/UD_English-EWT/en_ewt-um-train.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-train.conllu','./Data/UD_Turkish-IMST/tr_imst-um-train.conllu']
def getFeaturenTagVector(file, trainTagVector=None, trainFeat=None, trFeatIndex=None):
# for file in fileList:
	#print(file)
	wordDict={}
	wordDict1={}#=collections.defaultdict(list)
	tagDict={}
	features=[]
	with open(file) as fin:
		lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
	for nr,rows in enumerate(lines):
		word=str(rows[2])
		tagDim=str(rows[5])
		# validTag=True
		# if trainTagVector!=None and tagDim not in trainTagVector:
		# 	validTag=False
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

	# preFeat=defaultdict(int)
	# sufFeat=defaultdict(int)
	# # featureList=list(features)
	# featDict=defaultdict(int)
	# if trainFeat!=None:
	# 	diff=set(features)-set(trainFeat)
	# 	features=list(set(features)-diff)

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
				# print("word",w,i)
				# print("tag",t,k)
				# print(wordDict1[w])
				# print()
				tagVector[i][k]=1
	# print(tagVector.shape)
	return((wordDict,wordDict1,tagDict,features,trainVector,tagVector,featIndex))

def trainData(trainVector,tagVector,taglen,testWordVector, testTagVector):
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	# sess = tf.Session()
	initW = numpy.random.randn(901, taglen)
	W = tf.Variable(initW, dtype=tf.float64)
	# score = tf.matmul(x, W)
	# # if isinstance(test):#,numpy.ndarray):
	# # if Test:
	# # 	print("True")
	# # 	score=tf.matmul(x,test)
	# #y = tf.convert_to_tensor(tagVector[0])#,preferred_dtype=float64)
	# #x = tf.convert_to_tensor(trainVector)#,preferred_dtype=float64)
	# init = tf.global_variables_initializer()
	# sess.run(init)
	# # scVal = sess.run(score, {x : trainVector})
	# # print(scVal.shape)
	# prob = 1 / (1 + tf.exp(-score))
	# # prob = tf.clip_by_value(prob, 0.000001, 0.999999)
	# if (test):#,numpy.ndarray):
	# 	# testSess = tf.Session()
	# 	predictedProb = sess.run(prob, {x : trainVector, y : tagVector})
	# 	return(predictedProb)
	# logPY = y * tf.log(prob) + (1-y) *tf.log(1 - prob)
	# # logLikelihood = -tf.reduce_sum(logPY)
	# meanLL = -tf.reduce_mean(logPY)
	# # dw = tf.gradients(meanLL, [W,])
	#
	batch_size=32
	eta = .01
	loss_batch=[]
	# wValue = sess.run(W)
	# # print(type(wValue))
	# # exit()
	# train_op = tf.train.AdamOptimizer(0.01).minimize(meanLL)#,var_list=[W,])#,prob,score])
	# model = tf.global_variables_initializer()
	# # print(trainVector.shape)
	# # print(tagVector[:,0].shape)
	# # print(len(tagVector[0]))
	# # print(tagVector[1].shape)
	# sess.run(model)
	score = tf.matmul(x, W)
	prob = 1 / (1 + tf.exp(-score))
	# clipping the probability by both sides to prevent the underflow
	prob = tf.clip_by_value(prob, 0.000001, 0.999999)
	logPY = y * tf.log(prob) + (1-y) * tf.log(1 - prob)
	"""Cost function = average loss for each example per the definition"""
	# the mean of logPY is the cost function we are trying to minimize
	meanLL = -tf.reduce_mean(logPY)

	# The Gradient Descent Optimizer does the heavy lifting where the first parameter is the learning rate
	"""Leaning rate - needs continuous tuning to work well"""
	train_op = tf.train.AdamOptimizer(0.005).minimize(meanLL)
	# initialize values, create a session and run the model
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	"""定义一个变量表示算出来的cost_value"""
	#cost_value = sess.run(cost_func, {x:feat_matrix, y:tag_matrix})

	wValue = sess.run(W)

	for i in range(5):
		for index, offset in enumerate(range(0, tagVector.shape[0], batch_size)):
			# then divide the matrices by rows
			remaining = tagVector.shape[0] - offset
			# if the remaining rows are smaller than the batch size, only batch the remaining of the matrices
			if remaining < batch_size:
				x_batch, y_batch = trainVector[offset : offset + remaining], tagVector[offset : offset + remaining]
			else:
				x_batch, y_batch = trainVector[offset : offset + batch_size], tagVector[offset : offset + batch_size]
			loss, d = sess.run([meanLL,train_op], {x : x_batch, y : y_batch})
			print(loss)
			loss_batch.append(loss)
			wValue = sess.run(W)
	# EPOCH = 5
	# BATCH_SIZE = 32
	# TRAIN_DATASIZE,_= trainVector.shape
	# PERIOD = int(TRAIN_DATASIZE/BATCH_SIZE) #Number of iterations for each epoch
	#
	# for e in range(EPOCH):
	# 	idxs = numpy.random.permutation(TRAIN_DATASIZE) #shuffled ordering
	# 	X_random = trainVector[idxs]
	# 	Y_random = trainVector[idxs]
	# 	for i in range(PERIOD):
	# 		batch_X = X_random[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
	# 		batch_Y = Y_random[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
	# 		gradVal, mll=sess.run([meanLL,train_op],{x: batch_X, y:batch_Y})
	# 		print('Loss = ', gradVal)
	# 		loss_batch.append(gradVal)
	# for kl in range(len(tagVector[0])):
	# 	for i in range(1000):
	# 		subfeats, subtags = getNextBatch(trainVector, trainVector[:,kl], 32)
	# 		gradVal, mll = sess.run([train_op,meanLL],{x : subfeats, y : subtags})
	# 		if(i+1) % 100 == 0:
	# 			#print('Step #', str(i), 'W = ', str(sess.run(W)))
	# 			print('Loss = ', mll)
	# 			loss_batch.append(mll)
	# 			wValue = sess.run(W)
	#plt.plot(range(0, 420, 25), loss_batch, 'r--', label='Batch Loss for tagsets')
	#print(w_value)
	#print(sum(loss_batch)/float(len(loss_batch)))
	# x1 = numpy.linspace(0, len(tagVector[0]), len(tagVector[0]), endpoint=False)
	# plt.scatter(x1,loss_batch, marker='o', c='b')
	# plt.savefig(file+'Adam.png')
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	score = tf.matmul(x, wValue)
	prob = 1 / (1 + tf.exp(-score))
	dev_sess = tf.Session()
	predictedProb = dev_sess.run(prob, {x : testWordVector, y : testTagVector})
	return(predictedProb)

def main():
	trainFile='./Data/UD_English-EWT/en_ewt-um-train.conllu'
	testFile='./Data/UD_English-EWT/en_ewt-um-dev.conllu'
	trainWordDict,trainWordDictList,trainTagDict,trainFeat,trainWordVector,trainTagVector,featIndex=getFeaturenTagVector(trainFile)
	print(trainWordVector.shape)
	print(trainTagVector.shape)
	testWordDict,testWordDictList,testTagDict,testFeat,testWordVector,testTagVector,featIndex=getFeaturenTagVector(testFile,trainTagDict,trainFeat,featIndex)
	print(testWordVector.shape)
	print(testTagVector.shape)
	print(len(testTagVector[0]))
	# exit()
	predictedProb=trainData(trainWordVector,trainTagVector,len(trainTagDict),testWordVector,testTagVector)
	# x = tf.placeholder(tf.float64)
	# y = tf.placeholder(tf.float64)
	# score = tf.matmul(x, wValue)
	# prob = 1 / (1 + tf.exp(-score))
	# dev_sess = tf.Session()
	# predictedProb = dev_sess.run(prob, {x : testWordVector, y : testTagVector})
	# # predictedProb=trainData(testWordVector,testTagVector,len(testTagDict),True)
	# print(predictedProb.shape)
	# # print(len(predictedProb[0]))
	proposed_tag = defaultdict(list)
	# exit()
	correct=0
	"""Access through each row(word example) for its prob of possessing each tag dimension"""
	for word in testWordDict:
		# print(word)
		for tag in testTagDict:
			# print("%s,	%s",word,tag)
	#        print (predicted_prob[row][column])
			wordInd = testWordDict[word]
			# retrieving the tag associated with each column
			tagInd = testTagDict[tag]
			"""Once the prob is greater than 0.5, it is proposed to have that tag"""
			try:
				if predictedProb[wordInd][tagInd] > 0.5:
					# print(tag)
					proposed_tag[word].append(tag)
			except Exception as e:
				print(e)
				print(wordInd)
				print(tagInd)
				# exit()
		# print(len(proposed_tag[word]))

		actSet=set(testWordDictList[word])
		proposedSet=set(proposed_tag[word])
		if actSet==proposedSet:
			print("True1 Word:	",word[1])
			print("True Actual:	",testWordDictList[word])
			print("True Predicted:	",proposed_tag[word])
			correct+=1#len(actSet.intersection(proposedSet))
			print()
		else:
			print("Word:	",word)
			print("Actual:	",testWordDictList[word])
			print("Predicted:	",proposed_tag[word])
			print(actSet-proposedSet)
			print(proposedSet-actSet)

	print(correct)
	print(correct/len(testWordDictList))

	prop=0
	for word in testWordDictList:
		templist=testWordDictList[word]
		proposedTagList=proposed_tag[word]
		for t in templist:
			# print(t)
			if(t in proposedTagList):
				# print(t)
				prop+=1
	print(prop)
if __name__ == "__main__":
	main()
