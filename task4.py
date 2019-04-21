from __future__ import division
from collections import defaultdict
from collections import Counter
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import heapq
import operator
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib as plt
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
import string

class ContinueI(Exception):
	pass

def readData(file):
	sentLength=[]
	wordList=[]
	sentences=[]
	sentence_tags=[]
	with open(file) as f:
		 fin = f.readlines()
	i=0
	while i<len(fin):
		if(fin[i])=='\n':
			i+=1
			continue
		if fin[i][0:7]=='# text ':
			i+=1
			tempSent=[]
			tempTag=[]

			while(i<len(fin) and fin[i] is not '\n'):
				sentences.append(fin[i])
				line=fin[i]
				lines=line.split('\t')
				if lines[1]!='' and lines[5]!='':# and lines[1]!='.' and lines[5]!='_' and lines[1]!='?' and lines[1]!=',' and lines[1]!='!':
					tempSent.append(lines[1])
					tempTag.append(lines[5])
				i+=1
			wordList.append(tempSent)
			sentence_tags.append(tempTag)
			sentLength.append(len(tempSent))
		else:
			i+=1
	lineLength=np.array(sentLength)
	# print(lineLength[0])
	mlen=np.percentile(lineLength,95)
	# for n in range(len(wordList)):
	# 	print(wordList[n],sentence_tags[n])
	# exit()
	return((wordList,sentence_tags,sentences,mlen))

def encodeSeq(sequences, categories):
	encodings= []
	for s in sequences:
		enc = []
		for item in s:
			enc.append(np.zeros(categories))
			enc[-1][item] = 1.0
		encodings.append(enc)
	return np.array(encodings)

def trainModel(trainSentences_X,trainTags_y,tag2index,word2index,MAX_LENGTH):
	HIDDEN_DIM = 96
	model = keras.models.Sequential()
	"""output_dim is 64 hidden units"""
	"""
	Embedding Layer:
	vocabulary size of len(word_set)=10002 words (10002 integer encodings)
	output is the vector embedding of words in 64 dimensions
	the input_length is the constant max_sent_len that is already calculated
	"""
	num_tags = len(tag2index)
	model.add(keras.layers.Embedding(len(word2index), HIDDEN_DIM, input_length=MAX_LENGTH))

	model.add(keras.layers.TimeDistributed(keras.layers.Dense(HIDDEN_DIM, activation="sigmoid")))

	model.add(keras.layers.LSTM(HIDDEN_DIM, activation='tanh', return_sequences=True))

	"""Optional Tuninig: Adding hidden Dense layers and Batch Normalization between Dense layers"""
	hidden = 1
	for ii in range(hidden):
		model.add(keras.layers.Dense(HIDDEN_DIM, activation="relu"))
		"""Batch Normalization - with 'relu' activation"""
		model.add(keras.layers.BatchNormalization())

	"""Dense layer also outputs num_tags dimensions"""
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_tags, activation="sigmoid")))
	"""Finally compile the optimizer and fit the data"""
	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
	callbacks = [EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=2)]
	model.summary()
	model.fit(trainSentences_X, trainTags_y, epochs=10, batch_size=128,verbose=1)
	return(model)

def trainLSTM(trainSentences_X,trainTags_y,testSentences_X,testTags_y,tag2index,word2index,MAX_LENGTH):
	model = Sequential()
	model.add(InputLayer(input_shape=(MAX_LENGTH, )))
	model.add(Embedding(len(word2index), 64))
	model.add(LSTM(64, return_sequences=True))
	model.add(TimeDistributed(Dense(len(tag2index))))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer=Adam(0.001),metrics=['accuracy'])
	model.fit(trainSentences_X, trainTags_y, batch_size=128, epochs=3)#, validation_split=0.2)
	predicted=model.predict(testSentences_X)
	scores = model.evaluate(testSentences_X,testTags_y)
	print(f"{model.metrics_names[1]}: {scores[1] * 100}")
	return(predicted)


def main():
	#testFileList=['./Data/UD_Spanish-AnCora/es_ancora-um-dev.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-dev.conllu','./Data/UD_Russian-GSD/ru_gsd-um-dev.conllu','./Data/UD_English-EWT/en_ewt-um-dev.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-dev.conllu','./Data/UD_Turkish-IMST/tr_imst-um-dev.conllu']
	#trainFileList=['./Data/UD_Spanish-AnCora/es_ancora-um-train.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-train.conllu','./Data/UD_Russian-GSD/ru_gsd-um-train.conllu','./Data/UD_English-EWT/en_ewt-um-train.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-train.conllu','./Data/UD_Turkish-IMST/tr_imst-um-train.conllu']
	testFileList=['Data/UD_English-EWT/en_ewt-um-dev.conllu']
	trainFileList=['Data/UD_English-EWT/en_ewt-um-train.conllu']

	trainRet=readData(trainFileList[0])
	trainSentences=trainRet[0]
	MAX_LENGTH_train=int(trainRet[3])
	# print(MAX_LENGTH_train)
	# print(trainSentences[1])
	trainTags=trainRet[1]
	# print(trainTags[1])
	testRet=readData(testFileList[0])
	testSentences=testRet[0]
	# print(len(testSentences[0]))
	# print(len(testSentences[0][0]))
	# exit()
	testTags=testRet[1]
	# print(testTags[0])
	# print(testTags[0][0])
	# exit()
	MAX_LENGTH_test=int(testRet[3])
	words, tags = set(), set()
	wfreq=defaultdict(int)
	for s in trainSentences:
		for w in s:
			if w not in wfreq:
				wfreq[w]=1
			else:
				wfreq[w]+=1
	fw=Counter(wfreq)
	freqWords=fw.most_common(10000)
	for ts in trainTags:
		for t in ts:
			ti=t.split(';')
			for td in ti:
				tags.add(td)
	# print(len(tags))
	word2index = {w[0]: i + 2 for i, w in enumerate(freqWords)}
	word2index['PAD'] = 0  # The special value used for padding
	word2index['UNK'] = 1  # The special value used for OOVs

	tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
	tag2index['PAD'] = 0

	index2tag={}
	for t in tag2index:
		index2tag[tag2index[t]]=t

	print(len(word2index))
	print(len(tag2index))
	trainSentences_X=np.zeros((len(trainSentences), MAX_LENGTH_train))
	testSentences_X=np.zeros((len(testSentences), MAX_LENGTH_train))
	trainTags_y=np.zeros((len(trainSentences), MAX_LENGTH_train,len(tag2index)))
	testTags_y = np.zeros((len(testSentences), MAX_LENGTH_train,len(tag2index)))
	print(trainSentences_X.shape)
	print(trainTags_y.shape)
	print(testSentences_X.shape)
	print(testTags_y.shape)
	# exit()
	for i,sent in enumerate(trainSentences):
		wcount=0
		for nwords,word in enumerate(sent):
			if nwords<MAX_LENGTH_train:
				if word in word2index:
					trainSentences_X[i][nwords]=word2index[word]
				else:
					trainSentences_X[i][nwords]=word2index['UNK']
				tags=trainTags[i][nwords]
				tagSplit=tags.split(';')
				for tg in tagSplit:
					trainTags_y[i][nwords][tag2index[tg]]=1
					# print(i,nwords,tag2index[tg])
				wcount=nwords
		for wc in range(wcount,MAX_LENGTH_train):
			trainSentences_X[i][wc]=word2index['PAD']
	# print(trainSentences_X)
	# print(trainTags_y)

	testSentArray=defaultdict(str)
	testWords=defaultdict(int)
	testTagsDict=defaultdict(set)
	newL=[]
	for i,sent in enumerate(testSentences):
		wcount=0
		for nwords,word in enumerate(sent):
			if nwords<MAX_LENGTH_train:
				if word not in testWords:
					testWords[word]=1
				else:
					testWords[word]+=1
				if word in word2index:
					testSentences_X[i][nwords]=word2index[word]
					testSentArray[(i,nwords)]=word
				else:
					testSentences_X[i][nwords]=word2index['UNK']
					testSentArray[(i,nwords)]='UNK'
				tags=testTags[i][nwords]
				tagSplit=tags.split(';')
				testTagsDict[word].add(tuple(tagSplit))
				for tg in tagSplit:
					testTags_y[i][nwords][tag2index[tg]]=1
				wcount=nwords
		newL.append(wcount)
		for wc in range(wcount,MAX_LENGTH_train):
			testSentences_X[i][wc]=word2index['PAD']
			testSentArray[(i,wc)]='PAD'

	sampleWordSet = set()
	allCount = collections.defaultdict(int)
	"""Select words that occurs above 10 times, have more than _ tag vector, and the most frequent vector accounts for < 70%"""
	for word in testWords:
		temp = collections.Counter(testTagsDict[word])
		allCount[word] = len(list(temp.elements()))
		if testWords[word] > 10:
			if len(testTagsDict[word]) > 6:
				most_common = temp.most_common(1)[0][1]
				percentage = most_common / allCount[word]
				if percentage < 0.7:
					sampleWordSet.add(word)
					# print("Word: " + word)

	sampleSentSet = collections.defaultdict(set)
	for word in sampleWordSet:
		count = 0
		# print("10 context sentences for: " + word)
		for i,sent in enumerate(testSentences):
			if word in sent:
				sampleSentSet[word].add(i)
				# print(' '.join(sentence[sent_id]))
				count += 1
				if count >= 10:
					break

	print(len(sampleWordSet))


	# for t in (testSentArray):
	# 	print(t)
	# exit()
	# pred=trainLSTM(trainSentences_X,trainTags_y,testSentences_X,testTags_y,tag2index,word2index,MAX_LENGTH_train)
	model=trainModel(trainSentences_X,trainTags_y,tag2index,word2index,MAX_LENGTH_train)
	predicted=model.predict(testSentences_X)
	# discretePred=predicted>0.5
	# print(discretePred)
	# exit()
	proposed =defaultdict(set)

	newlist=[]
	word=''
	tag=''
	continue_outer= ContinueI()
	for sentInd in range(len(testSentences_X)):
		# if(sentInd==25):
		# 	break
		for wordInd in range(MAX_LENGTH_train):
			# print(testSentArray[(sentInd,wordInd)])
			# if (sentInd,wordInd) not in list(testSentArray.keys()):
				# print(testSentArray[(sentInd,wordInd)])
				# r/aise continue_outer
			try:
				for tagInd in range(len(tag2index)):
					word = testSentArray[(sentInd,wordInd)]
					if word!='UNK' and word!='PAD':
						tag = index2tag[tagInd]


						# newlist.append(predicted[sentInd][word2index[word]][tagInd])
						"""Once the prob is greater than 0.5, it is proposed to have that tag"""
						if predicted[sentInd][wordInd][tagInd] > 0.5:#and tag not in proposed[sentInd, wordInd, word]:
							proposed[sentInd, wordInd, word].add(tag)

						# if predicted[sentInd][word2index[word]][tagInd] > 0.5:
						# 	act=testTags[sentInd][wordInd]
							# print("yo")
							# print("Word: "+str(word)+"	PTag: "+str(proposed[sentInd, wordInd, word])+"	Actual:"+str(act))
					else:

						if predicted[sentInd][wordInd][tagInd] > 0.5:
							# print("Hatrappa")
							act=testTags[sentInd][wordInd]
							# print("Wor/d: "+str(word)+"	PTag: "+str(proposed[sentInd, wordInd, word])+"	Actual:"+str(act))

			except Exception as E:
				print(sentInd,wordInd,tagInd)
				# print(wordInd)
				# print(tagInd)
				# print("hatroooo")
				print(E)
				continue
	# print(len(newlist))
	# exit()
	pls=0
	for sentInd in range(len(testSentences_X)):
		for wordInd in range(MAX_LENGTH_train):
			# for tagInd in range(len(tag2index)):
			word = testSentArray[(sentInd,wordInd)]
			if word!='UNK' and word!='PAD':
				act=testTags[sentInd][wordInd]
				act=act.split(';')
				if set(act)==proposed[sentInd, wordInd, word]:
					# print("yo")
					pls+=1
					# print("Word: "+str(word)+"	PTag: "+str(proposed[sentInd, wordInd, word])+"	Actual:"+str(act))
				# else:
					# print("hatroo")
					# print(word)
					# print(act)
					# print(proposed[sentInd, wordInd, word])
	print(pls/sum(newL))
	# exit()

	ctd=0
	count=0
	for sentInd in range(len(testSentences_X)):
		for wordInd in range(MAX_LENGTH_train):
			word = testSentArray[(sentInd,wordInd)]
			if word in sampleWordSet and sentInd in sampleSentSet[word]:
				tags=(testTags[sentInd][wordInd]).split(';')
				current_sentence = ' '.join(testSentences[sentInd])
				print("WORD INVOLVED: " + word + "!!!!!!!!")
				print(current_sentence)
				print("Position in the sentence: " + str(wordInd))
				print("Proposed: " + str(proposed[sentInd, wordInd, word]))
				print("Actual: " +str(tags))
				print('------------------------------------------------')
				if set(tags) in proposed[sentInd, wordInd, word]:
					ctd+=1
					# print("Actual: "+str(testTags[sentInd][wordInd])+"Proposed: "+str(proposed[sentInd, wordInd, word]))
				if set(tags)== proposed[sentInd, wordInd, word]:
					count+=1
					# print("Actual: "+str(testTags[sentInd][wordInd])+" Proposed: "+str(proposed[sentInd, wordInd, word]))

	print("Counted	"+str(count))
	print(count/len(sampleWordSet))
	# for s in trainTags:
	# 	trainTags_y.append([tag2index[t] for t in s])
	#
	# for s in testTags:
	# 	try:
	# 		testTags_y.append([tag2index[t] for t in s])
	# 	except:
	# 		pass

	# print((trainTags_y[1][0]))
	# print(testTags_y.shape)
	# trainSentences_X = pad_sequences(trainSentences_X, maxlen=MAX_LENGTH, padding='post')
	# testSentences_X = pad_sequences(testSentences_X, maxlen=MAX_LENGTH, padding='post')
	# trainTags_y = pad_sequences(trainTags_y, maxlen=MAX_LENGTH, padding='post')
	# print(trainSentences_X.shape)
	# print(trainTags_y.shape)
	# testTags_y = pad_sequences(testTags_y, maxlen=MAX_LENGTH, padding='post')
	# print(testTags_y.shape)





if __name__ == "__main__":
	main()
