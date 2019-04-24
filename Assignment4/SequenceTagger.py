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
	fin=[]
	with open(file) as f:
		 inputLines = f.readlines()
	for i,lines in enumerate(inputLines):
		if '#' in lines:
			if '# sent_id ' not in lines and '# text ' not in lines:
				continue
			else:
				fin.append(lines)
		else:
			fin.append(lines)
	f.close()


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
				if lines[1]!='' and lines[5]!='':
					tempSent.append(lines[1])
					tempTag.append(lines[5])
				i+=1
			wordList.append(tempSent)
			sentence_tags.append(tempTag)
			sentLength.append(len(tempSent))
		else:
			i+=1
	lineLength=np.array(sentLength)
	mlen=np.percentile(lineLength,95)
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

def trainModel(trainSentences_X,trainTags_y,tag2index,word2index,MAX_LENGTH,HIDDEN_DIM,hidden,sanskrit=False):

	model = keras.models.Sequential()
	num_tags = len(tag2index)
	model.add(keras.layers.Embedding(len(word2index), HIDDEN_DIM, input_length=MAX_LENGTH))

	model.add(keras.layers.TimeDistributed(keras.layers.Dense(HIDDEN_DIM, activation="sigmoid")))

	model.add(keras.layers.LSTM(HIDDEN_DIM, activation='tanh', return_sequences=True))

	for ii in range(hidden):
		model.add(keras.layers.Dense(HIDDEN_DIM, activation="relu"))
		model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_tags, activation="sigmoid")))
	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
	callbacks = [EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=2)]
	model.summary()
	if sanskrit:
		model.fit(trainSentences_X, trainTags_y, epochs=100, batch_size=128,verbose=1)
	else:
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
	f.write(f"{model.metrics_names[1]}: {scores[1] * 100}")
	return(predicted)

def printEvalMetrics(actual,matched,proposed,tags,fileName,bl=False):
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
		for t in feat:
			if t not in matched:
				P = 1.0
				R = 0.0
			else:
				P = matched[t] / proposed[t]
				R = matched[t] / actual[t]
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
	dims=[96]
	layers=[1]
	for li in range(len(lang)):
		for di in dims:
			for hi in layers:
				print(lang[li])
				trainRet=readData(trainFiles[li])
				trainSentences=trainRet[0]
				MAX_LENGTH_train=int(trainRet[3])
				trainTags=trainRet[1]
				testRet=readData(testFiles[li])
				testSentences=testRet[0]
				testTags=testRet[1]
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

				word2index = {w[0]: i + 2 for i, w in enumerate(freqWords)}
				word2index['PAD'] = 0  # The special value used for padding
				word2index['UNK'] = 1  # The special value used for OOVs

				tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
				tag2index['PAD'] = 0

				index2tag={}
				for t in tag2index:
					index2tag[tag2index[t]]=t

				# f.write(len(word2index))
				# f.write(len(tag2index))
				trainSentences_X=np.zeros((len(trainSentences), MAX_LENGTH_train))
				testSentences_X=np.zeros((len(testSentences), MAX_LENGTH_train))
				trainTags_y=np.zeros((len(trainSentences), MAX_LENGTH_train,len(tag2index)))
				testTags_y = np.zeros((len(testSentences), MAX_LENGTH_train,len(tag2index)))
				# f.write(trainSentences_X.shape)
				# f.write(trainTags_y.shape)
				# f.write(testSentences_X.shape)
				# f.write(testTags_y.shape)
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
							wcount=nwords
					for wc in range(wcount,MAX_LENGTH_train):
						trainSentences_X[i][wc]=word2index['PAD']

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
				for word in testWords:
					temp = collections.Counter(testTagsDict[word])
					allCount[word] = len(list(temp.elements()))
					if testWords[word] > 10:
						if len(testTagsDict[word]) > 6:
							most_common = temp.most_common(1)[0][1]
							percentage = most_common / allCount[word]
							if percentage < 0.7:
								sampleWordSet.add(word)

				f=open(lang[li]+str(di)+str(hi)+"_context","w+")
				sampleSentSet = collections.defaultdict(set)
				for word in sampleWordSet:
					count = 0
					f.write("10 context sentences for: " + word+"\n")
					for i,sent in enumerate(testSentences):
						if word in sent:
							sampleSentSet[word].add(i)
							f.write(' '.join(sent)+"\n")
							count += 1
							if count >= 10:
								break
					f.write("____________________________________________")

				f.close()

				# pred=trainLSTM(trainSentences_X,trainTags_y,testSentences_X,testTags_y,tag2index,word2index,MAX_LENGTH_train)
				if lang[li]=='Sanskrit':
					model=trainModel(trainSentences_X,trainTags_y,tag2index,word2index,MAX_LENGTH_train,di,hi,True)
				else:
					model=trainModel(trainSentences_X,trainTags_y,tag2index,word2index,MAX_LENGTH_train,di,hi)
				predicted=model.predict(testSentences_X)
				proposed =defaultdict(set)

				newlist=[]
				word=''
				tag=''
				continue_outer= ContinueI()
				for sentInd in range(len(testSentences_X)):
					for wordInd in range(MAX_LENGTH_train):
						try:
							for tagInd in range(len(tag2index)):
								word = testSentArray[(sentInd,wordInd)]
								if word!='UNK' and word!='PAD':
									tag = index2tag[tagInd]
									if predicted[sentInd][wordInd][tagInd] > 0.5:
										proposed[sentInd, wordInd, word].add(tag)
								else:

									if predicted[sentInd][wordInd][tagInd] > 0.5:
										act=testTags[sentInd][wordInd]
						except Exception as E:
							# f.write(sentInd,wordInd,tagInd)
							# f.write(E)
							continue

				correct=0
				actual=defaultdict(int)
				matched=defaultdict(int)
				proposedSend=defaultdict(int)
				for sentInd in range(len(testSentences_X)):
					for wordInd in range(MAX_LENGTH_train):
						word = testSentArray[(sentInd,wordInd)]
						if word!='UNK' and word!='PAD':
							act=testTags[sentInd][wordInd]
							act=act.split(';')
							for t in act:
								actual[tag]+=1
							if set(act)==proposed[sentInd, wordInd, word]:
								for t in act:
									matched[tag]+=1
								correct+=1

				f=open(lang[li]+str(di)+str(hi),"w+")
				f.write("Accuracy:  "+str((correct/sum(newL)))+"\n")
				f.close()

				for p in proposed:
					for tag in proposed[p]:
						proposedSend[tag]+=1
				printEvalMetrics(actual,matched,proposedSend,testTags,lang[li]+str(di)+str(hi))
				ctd=0
				count=0
				f=open(lang[li]+str(di)+str(hi)+"_context","a+")
				for sentInd in range(len(testSentences_X)):
					for wordInd in range(MAX_LENGTH_train):
						word = testSentArray[(sentInd,wordInd)]
						if word in sampleWordSet and sentInd in sampleSentSet[word]:
							tags=(testTags[sentInd][wordInd]).split(';')
							current_sentence = ' '.join(testSentences[sentInd])
							f.write("WORD:  " + word+"\n")
							f.write("Occurance:  "+current_sentence+"\n")
							f.write("Proposed: " + str(proposed[sentInd, wordInd, word])+"\n")
							f.write("Actual: " +str(tags)+"\n")
							f.write('___________________________________________')
							if set(tags) in proposed[sentInd, wordInd, word]:
								ctd+=1
								# f.write("Actual: "+str(testTags[sentInd][wordInd])+"Proposed: "+str(proposed[sentInd, wordInd, word]))
							if set(tags)== proposed[sentInd, wordInd, word]:
								count+=1
								# f.write("Actual: "+str(testTags[sentInd][wordInd])+" Proposed: "+str(proposed[sentInd, wordInd, word]))

				f.write("Sample set accuracy:  ")
				try:
					f.write(str(count/len(sampleWordSet))+"\n")
				except:
					f.write(str(0.0))
				f.close()




if __name__ == "__main__":
	main()
