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


class colors:
	ok = '\033[92m'
	fail = '\033[91m'
	close = '\033[0m'

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
		# charSet['UNK']=-999
		# charSetTarget['UNK']=-999
		# with open(filename) as f:
		f=open(filename)
		for line in f:
			# print(line)
			is_data = re.search("^\d.*", line)
			if is_data is not None:
				raw_data = is_data.group(0)
				data = raw_data.split("\t")
		# print(line)
				word=str(data[1])
				chs.append(word)
				# print(word)
				lemma=str(data[2])
				cht.append(lemma)
				# print(lemma)
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

		# print(len(chs))
		# print(len(cht))
		wordLengthSource=np.array(wordLengthSource)
		lemmaLengthTarget=np.array(lemmaLengthTarget)
		MAX_LEN_Source=int(np.percentile(wordLengthSource,95))
		MAX_LEN_Target=int(np.percentile(lemmaLengthTarget,95))
		for i in range(len(sourceList)):
			# sourceTarget[sourceList[i]]=[]
			sourceTarget[sourceList[i]]=targetList[i]
		print(len(chs))
		print(len(cht))
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
				for c in range(charc,Source_MAX_LEN_ ):
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
				for c in range(charc,Target_MAX_LEN_):
					outputMatrix[wc][c][ characterSet_['PADD']]=1

		return((inputMatrix,outputMatrix))

	def decode(self,x,indices_char):
		x = x.argmax(axis=-1)
		return ''.join(indices_char[x] for x in x)

def argMax(array):
    ind, mx = 0, 0
    for i, value in enumerate(array):
        if value > mx:
            ind, mx = i, value
    return ind, mx

def main():
	sig=Lemmatizer()
	trainData,characterSet,Source_MAX_LEN_train,Target_MAX_LEN_train,chs,cht=sig.getData('/home/karthik/SP19/CompLing/SigMorph/Assignment4/Data/UD_English-EWT/en_ewt-um-train.conllu')
	testData, testCharacterSet,Source_MAX_LEN_test,Target_MAX_LEN_test,chtest,chstest=sig.getData('/home/karthik/SP19/CompLing/SigMorph/Assignment4/Data/UD_English-EWT/en_ewt-um-dev.conllu',False)
	trainInput,trainOutput=sig.getMatrices(trainData,characterSet,Source_MAX_LEN_train,Target_MAX_LEN_train,chs,cht)
	testInput,testOutput=sig.getMatrices(testData, characterSet,Source_MAX_LEN_test,Target_MAX_LEN_test,chtest,chstest)

	ind2Char={}
	for w in characterSet:
		ind2Char[characterSet[w]]=w
	# print(len(trainData))
	# exit()
	# print(characterSet['PADD'])
	# print(ind2Char[0])

	# a=set(characterSet.keys())
	# b=set(ind2Char.values())
	# print(a.difference(b))
	# print(b.difference(a))
	# print(len(characterSet))
	# print(len(ind2Char))
	# print(trainInput.shape)
	# print(trainOutput.shape)
	# exit()
	RNN = keras.layers.LSTM
	HIDDEN_SIZE = 32
	BATCH_SIZE = 32
	LAYERS = 3

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

	# devwords=list()
	# devlemma=list(testData.values())
	pred_chars = collections.defaultdict(list)
	proposed_lemma = collections.defaultdict(list)
	# for iteration in range(1,10):
	# 	print()
	# 	print('Iteration', iteration)
	model.fit(trainInput, trainOutput,
			  batch_size=BATCH_SIZE,
			  epochs=10)
	prediction = model.predict(testInput)

	correct=0
	for ind in range(len(chtest)):
		# ind = np.random.randint(0, len(chtest))
		dword=chtest[ind]
		dlem=chstest[ind]
		# rowx, rowy = testInput[np.array([ind])], testOutput[np.array([ind])]
		for ch in range (0,Source_MAX_LEN_test):
			ch_ind,preds = argMax(prediction[ind][ch])
			pred_ch = ind2Char[ch_ind]
			pred_chars[ind, dword].append(pred_ch)
	# print(preds.shape)
	# q = sig.decode(rowx[0],ind2Char)
	# correct = sig.decode(rowy[0],ind2Char)
	# guess = sig.decode(preds[0], ind2Char)
	# print('Q', q, end=' ')
	# print('T', correct, end=' ')
	# if correct == guess:
	# 	print(colors.ok + '☑' + colors.close, end=' ')
	# else:
	# 	print(colors.fail + '☒' + colors.close, end=' ')
	# print(guess)
		depadding = (("".join(pred_chars[ind, dword])).replace('PADD', '')).strip()
		proposed_lemma[ind, dword] = depadding
		print("Word: "+str(dword)+"		 Actual: "+	str(dlem)+"		Predicted: "+str(proposed_lemma[ind, dword]))
		if(str(dlem)==str(proposed_lemma[ind, dword])):
			correct+=1

	print(correct/len(chtest))
if __name__ == "__main__":
	main()
