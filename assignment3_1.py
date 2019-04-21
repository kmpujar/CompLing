from __future__ import division
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy
import heapq
import operator
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from sklearn.manifold import TSNE
import matplotlib as plt

def vectorize(file):
	#print(file)
	wordDict={}
	wordDict1={}#=collections.defaultdict(list)
	tagDict={}
	features=[]
	with open(file) as fin:
		lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
	l=0
	for rows in lines:
		l+=1
		word=str(rows[2])
		tagDim=str(rows[5])
		if word not in wordDict1:
			wordDict1[word]=list()
		wordDict1[word].append(tagDim)
		if word not in wordDict:
			wordLen=len(word)
			#k=0
			#while(k<3):
			#	k+=1
			if(wordLen>0):
				features.append("PRE"+(word[0:1]))
				features.append("SUF"+(word[wordLen-1]))
			if(wordLen>1):
				features.append("PRE"+(word[0:2]))
				features.append("SUF"+(word[wordLen-2:]))
			if(wordLen>2):
				features.append("PRE"+(word[0:3]))
				features.append("SUF"+(word[wordLen-3:]))
			if word[wordLen-1:wordLen]=='न्' or word[wordLen-1:wordLen]=='त्' or word[wordLen-1:wordLen]=='म्' or word[wordLen-1:wordLen]=='द्':
				features.append(word[wordLen-1:wordLen])
			#featuresPre.add(word[0:k])
			# featuresPre.add(word[0:2])
			# featuresPre.add(word[0:3])
			#featuresSuf.add(word[wordLen-k:wordLen])
			# featuresSuf.add(word[wordLen-2:wordLen])
			# featuresSuf.add(word[wordLen-3:wordLen])
			wordDict[word]=len(wordDict)+1
		if tagDim not in tagDict:
			tagDict[tagDim]=len(tagDict)+1
	preFeat=defaultdict(int)
	sufFeat=defaultdict(int)
	featureList=list(features)
	featDict=defaultdict(int)

	counter=Counter(features)
	freqFeat=dict(counter.most_common(300))
	trainVector=numpy.zeros((len(wordDict), 300))
	#print(len(wordDict))
	i=-1
	k=-1
	for f in freqFeat:
		i+=1
		0
		k=-1
		for d in wordDict:
			wordLen=len(str(d))
			k+=1
			if len(f)==1 and d[wordLen-1:wordLen]==f:
				trainVector[k][i]=1
			if len(f)==4 and wordLen>0:
				if(f[0:3]=='PRE' and f[3:]==d[0]):
					trainVector[k][i]=1
				if(f[0:3]=='SUF' and f[3:]==d[wordLen-1]):
					trainVector[k][i]=1
			elif len(f)==5 and wordLen>1:
				if(f[0:3]=='PRE' and f[3:]==d[0:2]):
					trainVector[k][i]=1
				elif(f[0:3]=='SUF' and f[3:]==d[wordLen-2]):
					trainVector[k][i]=1
			elif len(f)==6 and wordLen>2:
				if(f[0:3]=='PRE' and f[3:]==d[0:3]):
					trainVector[k][i]=1
				elif(f[0:3]=='SUF' and f[3:]==d[wordLen-3]):
					trainVector[k][i]=1
	# print(trainVector)
	columnNames=set()
	rowNames={}
	#print((tagDict))
	tagVec=numpy.zeros((len(wordDict1), len(tagDict)))
	i=-1
	k=-1
	for w in wordDict1:
		i+=1
		rowNames[i]=w
		k=-1
		for t in tagDict:
			if(i==0):
				columnNames.add(t)
			k+=1
			if(t in wordDict1[w]):
				tagVec[i][k]=1
	#print(len(columnNames))
	#print(tagVec.shape)
	tagDF=pd.DataFrame(tagVec)
	tagDF.columns=list(columnNames)
	#tagDF=tagDF.rename(rowNames)
	#print(tagDF)

	return(trainVector,tagDF,rowNames)

def kerasMaxEnt(Xmatrix,Ymatrix,x_test):
	retTag=pd.DataFrame()
	# for kl in range(ymatrix.shape[1]):
	# 	Ymatrix=ymatrix.iloc[:,kl]
	model = keras.models.Sequential()
	model.add(Dense(Ymatrix.shape[1], input_shape=(300,),activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	model.fit(Xmatrix, Ymatrix,  batch_size=32,epochs=10,verbose=1)
	predictions = model.predict(x_test)
	discretePreds = predictions > .5
	#discretePreds=1*numpy.hstack(discretePreds)
		# retTag[ymatrix.columns[kl]]=discretePreds
	print(predictions.shape)
	print(type(predictions))
	df_1 = pd.DataFrame(discretePreds,columns=list(Ymatrix))
	print(df_1.shape)
	return df_1

def multiLayer(Xmatrix,Ymatrix,x_test,layers,batchNorm,isTSNE):
	retTag=pd.DataFrame()
	#for kl in range(trainV.shape[1]):
	# model = keras.models.Sequential()
	# model.add(Dense(128, input_shape=(300,),activation="sigmoid"))
	# model.add(keras.layers.normalization.BatchNormalization())
	# model.add(Dense(128, input_shape=(300,),activation="sigmoid"))
	# model.add(Dense(Ymatrix.shape[1], input_shape=(300,),activation="sigmoid"))
	for ii in range(0,layers-1):
		model.add(Dense(128, input_shape=(300,),activation="relu"))
		if(ii<layers and batchNorm):
			model.add(BatchNormalization())
	model.add(Dense(Ymatrix.shape[1]), input_shape=(300,),activation="sigmoid"))#, input_shape=(300,),activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	model.fit(Xmatrix, Ymatrix,  batch_size=32,epochs=1,verbose=1)
	predictions = model.predict(x_test)
	discretePreds = predictions > .5
	#discretePreds=1*numpy.hstack(discretePreds)
	if isTSNE and layers==3:
		get_2nd_layer_output = K.function([model.layers[0].input],
	                                  [model.layers[2].output])
		sec_op = get_2nd_layer_output([x_test])[0]
		#sec_op=sec_op[numpy.random.choice(sec_op.shape[0], 200, replace=False)]

		layer_of_interest=2
		intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])

		intermediates = []
		color_intermediates = []
		for i in range(200):
		    output_class = np.argmax(y_test.iloc[i,:].values)
		    intermediate_tensor = intermediate_tensor_function([X.iloc[i,:].values.reshape(1,-1)])[0]
		    intermediates.append(intermediate_tensor[0])
		    if(output_class == 0):
		        color_intermediates.append("#0000ff")
		    else:
		        color_intermediates.append("#ff0000")
		tsne = TSNE(n_components=2, random_state=0)
		intermediates_tsne = tsne.fit_transform(intermediates)
		plt.figure(figsize=(8, 8))
		plt.scatter(x = intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
		plt.show()
	df_1 = pd.DataFrame(discretePreds,columns=list(Ymatrix))
	#print(df_1.shape)
	return df_1

def main():
	testFileList=['./Data/UD_Spanish-AnCora/es_ancora-um-dev.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-dev.conllu','./Data/UD_Russian-GSD/ru_gsd-um-dev.conllu','./Data/UD_English-EWT/en_ewt-um-dev.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-dev.conllu','./Data/UD_Turkish-IMST/tr_imst-um-dev.conllu']
	trainFileList=['./Data/UD_Spanish-AnCora/es_ancora-um-train.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-train.conllu','./Data/UD_Russian-GSD/ru_gsd-um-train.conllu','./Data/UD_English-EWT/en_ewt-um-train.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-train.conllu','./Data/UD_Turkish-IMST/tr_imst-um-train.conllu']
	#testFileList=['../Data/UD_Turkish-IMST/tr_imst-um-dev.conllu']
	#trainFileList=['../Data/UD_Turkish-IMST/tr_imst-um-train.conllu']
	for i in range(len(testFileList)):
		trainV, tagV, trainWordList=vectorize(trainFileList[i])
		testV, testTagV, testWordList=vectorize(testFileList[i])
		#kerasMaxEnt(trainV,tagV,testV)
		for k in range(1,4):
			if('English' in trainFileList[i] or 'Sanskrit' in testFileList[i]):
				multiLayer(trainV,tagV,testV,k,True)
		break
if __name__ == "__main__":
	main()
