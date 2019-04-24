from __future__ import division
from collections import defaultdict
from collections import Counter
import regex as re
import pandas as pd
import numpy as np
import string
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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib as plt
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

class BaseLine:
	def __init__(self, sourceTarget):
		self.words=list(sourceTarget.keys())
		self.lemmas=list(sourceTarget.values())

	def evalData(self):
		correct=0
		for i in range(len(self.words)):
			if self.words[i]==self.lemmas[i]:
				correct+=1
		return(correct/len(self.words))

class Lemmatizer:

	def getData(self,filename,trainSource=True):
		sourceTarget=defaultdict(list)
		sourceList=[]
		targetList=[]
		wordLengthSource=[]
		lemmaLengthTarget=[]
		charSet=defaultdict(int)
		charSet['PADD']=len(charSet)
		chs=[]
		cht=[]
		f=open(filename)
		for line in f:
			is_data = re.search("^\d.*", line)
			if is_data is not None:
				raw_data = is_data.group(0)
				data = raw_data.split("\t")
				word=str(data[1])
				chs.append(word)
				lemma=str(data[2])
				cht.append(lemma)
				sourceList.append(word)
				wordLengthSource.append(len(word))
				targetList.append(lemma)
				lemmaLengthTarget.append(len(lemma))
				if trainSource is True:
					for c in word:
						if c not in charSet:
							charSet[c]=len(charSet)
				if trainSource is True:
					for c in lemma:
						if c not in charSet:
							charSet[c]=len(charSet)

		wordLengthSource=np.array(wordLengthSource)
		lemmaLengthTarget=np.array(lemmaLengthTarget)
		MAX_LEN_Source=int(np.percentile(wordLengthSource,95))
		MAX_LEN_Target=int(np.percentile(lemmaLengthTarget,95))
		for i in range(len(sourceList)):
			sourceTarget[sourceList[i]]=targetList[i]
		return((sourceTarget,charSet,MAX_LEN_Source,MAX_LEN_Target,chs,cht))

	def getMatrices(self, Data, characterSet_,Source_MAX_LEN_,Target_MAX_LEN_,chs,cht):
		inputMatrix=np.zeros((len(chs),Source_MAX_LEN_ ,len(characterSet_)))
		outputMatrix=np.zeros((len(cht),Target_MAX_LEN_ ,len(characterSet_)))
		for wc,word in enumerate(chs):
			charc=0
			for cc,char in enumerate(word):
				if(cc<Source_MAX_LEN_) and char in characterSet_:
					try:
						inputMatrix[wc][cc][ characterSet_[char]]=1
						charc=cc
					except Exception as e:
						print(e)
			if charc<Source_MAX_LEN_ :
				for c in range(charc+1,Source_MAX_LEN_ ):
					inputMatrix[wc][c][ characterSet_['PADD']]=1
		for wc,word in enumerate(cht):
			charc=0
			for cc,char in enumerate(word):
				if(cc<Target_MAX_LEN_ ) and char in characterSet_:
					try:
						outputMatrix[wc][cc][ characterSet_[char]]=1
						charc=cc
					except Exception as e:
						print(e)
			if charc<Target_MAX_LEN_ :
				for c in range(charc+1,Target_MAX_LEN_):
					outputMatrix[wc][c][ characterSet_['PADD']]=1

		return((inputMatrix,outputMatrix))

	def decode(self,x,indices_char):
		x = x.argmax(axis=-1)
		return ''.join(indices_char[x] for x in x)

	def argMax(self,array):
	    ind, mx = 0, 0
	    for i, value in enumerate(array):
	        if value > mx:
	            ind, mx = i, value
	    return ind, mx

	def runLSTM(self,trainInput, trainOutput,testInput,Target_MAX_LEN_train,Source_MAX_LEN_train,characterSet,HIDDEN_SIZE,LAYERS,sans=False):
		RNN = keras.layers.LSTM
		PREDICTIONS=[]
		EPOCHS=10
		if sans:
			EPOCHS=100
		# RNN = keras.layers.LSTM
		# HIDDEN_SIZE = 32
		# BATCH_SIZE = 32
		# LAYERS = 3

		print('Build model...')
		model = keras.models.Sequential()
		# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
		# Note: In a situation where your input sequences have a variable length,
		# use input_shape=(None, num_feature).
		model.add(RNN(HIDDEN_SIZE, input_shape=(Source_MAX_LEN_train, len(characterSet))))
		# As the decoder RNN's input, repeatedly provide with the last output of
		# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
		# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
		model.add(keras.layers.RepeatVector(Target_MAX_LEN_train))
		# The decoder RNN could be multiple layers stacked or a single layer.
		for _ in range(LAYERS):
			# By setting return_sequences to True, return not only the last output but
			# all the outputs so far in the form of (num_samples, timesteps,
			# output_dim). This is necessary as TimeDistributed in the below expects
			# the first dimension to be the timesteps.
			model.add(RNN(HIDDEN_SIZE, return_sequences=True))

		# Apply a dense layer to the every temporal slice of an input. For each of step
		# of the output sequence, decide which character should be chosen.
		model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(characterSet), activation='softmax')))
		model.compile(loss='categorical_crossentropy',
					  optimizer='adam',
					  metrics=['accuracy'])
		model.summary()
		for ep in range(EPOCHS):
			history=model.fit(trainInput, trainOutput,
					  batch_size=128,
					  epochs=1,verbose=2)
			prediction = model.predict(testInput)
			PREDICTIONS.append(prediction)
		return(PREDICTIONS)


def main():
	trainFiles=['./Data/tr_imst-um-train.conllu','./Data/zh_gsd-um-train.conllu','./Data/sa_ufal-um-train.conllu','./Data/es_ancora-um-train.conllu']
	testFiles=['./Data/tr_imst-um-dev.conllu','./Data/zh_gsd-um-dev.conllu','./Data/sa_ufal-um-dev.conllu','./Data/es_ancora-um-dev.conllu']
	lang=[ 'Turkish', 'Chinese', 'Sanskrit', 'Spanish']
	HS=[32,64]
	nlayers=[2]
	for li in range(len(lang)):
		sig=Lemmatizer()
		trainData,characterSet,Source_MAX_LEN_train,Target_MAX_LEN_train,chs,cht=sig.getData(trainFiles[li])
		testData, testCharacterSet,Source_MAX_LEN_test,Target_MAX_LEN_test,chtest,chstest=sig.getData(testFiles[li],False)
		trainInput,trainOutput=sig.getMatrices(trainData,characterSet,Source_MAX_LEN_train,Target_MAX_LEN_train,chs,cht)
		testInput,testOutput=sig.getMatrices(testData, characterSet,Source_MAX_LEN_train,Target_MAX_LEN_train,chtest,chstest)
		ind2Char={}
		for w in characterSet:
			ind2Char[characterSet[w]]=w
		print(trainInput.shape)
		print(testInput.shape)
		print(trainOutput.shape)
		print(testOutput.shape)
		if li==0:
			base=BaseLine(testData)
			print("BaseLine accuracy:  ")
			print(base.evalData())
		for HIDDEN_SIZE in HS:
			for LAYERS in nlayers:
				EPOCHS=5
				if lang[li]=="Sanskrit":
					EPOCHS=50
				if lang[li]=="Spanish":
					EPOCHS=1
				RNN = keras.layers.LSTM
				print('Build model...')
				model = keras.models.Sequential()
				model.add(RNN(HIDDEN_SIZE, input_shape=(Source_MAX_LEN_train, len(characterSet))))
				model.add(keras.layers.RepeatVector(Target_MAX_LEN_train))
				for _ in range(LAYERS):
					model.add(RNN(HIDDEN_SIZE, return_sequences=True))
				model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(characterSet), activation='softmax')))
				model.compile(loss='categorical_crossentropy',
							  optimizer='adam',
							  metrics=['accuracy'])
				model.summary()
				f=open(lang[li]+"_"+str(HIDDEN_SIZE)+"_"+str(LAYERS),"w+")
				# for ep in range(EPOCHS):
				model.fit(trainInput, trainOutput,
						  batch_size=128,
						  epochs=EPOCHS,verbose=1)
				prediction = model.predict(testInput)
				wordc=0
				f.write("Results after "+str(EPOCHS+1)+" epochs...\n")
				predictedChars = collections.defaultdict(list)
				proposedLemma = collections.defaultdict(list)
				correct=0
				for ind in range(len(chtest)):
					dword=chtest[ind]
					dlem=chstest[ind]
					for ch in range (0,Target_MAX_LEN_train):
						ch_ind,preds = sig.argMax(prediction[ind][ch])
						pred_ch = ind2Char[ch_ind]
						predictedChars[ind, dword].append(pred_ch)
					depadding = (("".join(predictedChars[ind, dword])).replace('PADD', '')).strip()
					proposedLemma[ind, dword] = depadding
					if wordc<5:
						f.write("Word: "+str(dword)+"  Actual: "+	str(dlem)+"  Predicted: "+str(proposedLemma[ind, dword])+"\n")
						wordc+=1
						f.write("___________________________________________\n")
					if(str(dlem)==str(proposedLemma[ind, dword])):
						correct+=1
				f.write("Accuracy:  "+str(correct/len(chtest))+"\n")
				f.close()

if __name__ == "__main__":
	main()
