from __future__ import division
from collections import *
import pandas as pd
import numpy as np

class Interpolated:
	def __init__(self, file):
		self.featSet=set()
		self.allGramSet=[]
		self.wordTag={}
		self.wordSet=set()
		self.allFeaturePlus=[]
		self.allFeatureMinus=[]
		self.featurePlus={}
		self.featureMinus={}

		with open(file) as fin:
			lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
		for nr,rows in enumerate(lines):
			word=str(rows[1])
			self.wordSet.add(word)
			tagDim=str(rows[5])
			tags=tagDim.split(';')
			self.wordTag[(nr,word)]=tagDim
			for t in tags:
				self.featSet.add(t)

	def getnGrams(self,word,n):
		nullList=[]
		for null in range(1,n):
			nullList.append('**')
		padded = nullList+list(word)+ ['**']
		ngramlist = []
		for i in range(4, len(word)+5):
			ngramlist.append(tuple(padded[i-4 : i+1]))
		return(ngramlist)


	def getFeatureSets(self):

		oneGrams=defaultdict(int)
		for sentInd,word in self.wordTag:
			allGramList=[]
			tags=self.wordTag[(sentInd,word)].split(';')
			for i in range(5):
				allGramList.append(self.getnGrams(word,i+1))
				self.allFeaturePlus.append({})
				self.allFeatureMinus.append({})
				self.allGramSet.append(set())
			for g in allGramList[0]:
				oneGrams[g]+=1
			for feature in self.featSet:
				for ind,wgrams in enumerate(allGramList):
					for wg in wgrams:
						self.allGramSet[ind].add(wg)
						if feature in tags:
							if feature not in self.allFeaturePlus[ind]:
								self.allFeaturePlus[ind][feature]=defaultdict(int)
							self.allFeaturePlus[ind][feature][wg]+=1
						else:
							if feature not in self.allFeatureMinus[ind]:
								self.allFeatureMinus[ind][feature]=defaultdict(int)
							self.allFeatureMinus[ind][feature][wg]+=1
		return(oneGrams)

	def evalData(self):

		overallFeatPlus=[]
		overallFeatMinus=[]
		for i in range(5):
			overallFeatPlus.append({})
			overallFeatMinus.append({})

		for ind,featP in enumerate(self.allFeaturePlus):
			for feat in featP:
				for wg in featP[feat]:
					if feat not in overallFeatPlus[ind]:
						overallFeatPlus[ind][feat]=0
					overallFeatPlus[ind][feat] += featP[feat][wg]

		for ind,featM in enumerate(self.allFeatureMinus):
			for feat in featM:
				for wg in featM[feat]:
					if feat not in overallFeatMinus[ind]:
						overallFeatMinus[ind][feat]=0
					overallFeatMinus[ind][feat] += featM[feat][wg]

		return((overallFeatPlus,overallFeatMinus))

	def getUniProb(self):

		oneGramCount=self.getFeatureSets()
		uniProb = defaultdict(int)
		for gram in oneGramCount:
			uniProb[gram] = oneGramCount[gram] / len(oneGramCount)
		return(uniProb)

	def reduceGram(self,gram):
		if len(gram) == 2:
			reduced = (gram[1])
		else:
			reduced = gram[1:len(gram)]
		return reduced

	def argMax(self,num1, num2):
		if num1 >= num2:
			return num1
		else:
			return num2

	def argMin(self,num1, num2):
		if num1 <= num2:
			return num1
		else:
			return num2

	def absoluteDiscount(self,featurePlus, overallFeaturePlus, discount, prevGrams):

		num = self.argMax(0, featurePlus-discount)
		denm = overallFeaturePlus
		absDiscount = self.argMin(discount, featurePlus)
		return (num / denm) + ((absDiscount * prevGrams) / denm)

	def calcAbsoluteDiscount(self,trainObj):

		uniProb=trainObj.getUniProb()
		overallFeatPlus,overallFeatMinus=trainObj.evalData()
		featProbPlusList=[]
		featProbMinusList=[]
		featProbPlusList.append(uniProb)
		featProbMinusList.append(uniProb)
		for ind in range(1,len(overallFeatPlus)+1):
			featProbPlusList.append({})
			featProbMinusList.append({})
			for feat in self.allFeaturePlus[ind-1]:
				for gram in self.allGramSet[ind-1]:
					if feat not in featProbPlusList[ind]:
						featProbPlusList[ind][feat] = defaultdict(int)
					# print(featProbPlusList[ind][feat])
					# print(featProbPlusList[ind][feat][gram])
					# print(overallFeatPlus[ind-1])
					# print(overallFeatPlus[ind-1][feat])
					if gram not in self.allFeaturePlus[ind-1][feat]:
						if ind>1:
							featProbPlusList[ind][feat][gram] = self.absoluteDiscount(0, overallFeatPlus[ind-1][feat], 0.75, featProbPlusList[ind-1][feat][self.reduceGram(gram)])
						else:
							featProbPlusList[ind][feat][gram] = self.absoluteDiscount(0, overallFeatPlus[ind-1][feat], 0.75, featProbPlusList[ind-1][gram])
					else:
						if ind>1:
							featProbPlusList[ind][feat][gram] = self.absoluteDiscount(trainObj.allFeaturePlus[ind-1][feat][gram], overallFeatPlus[ind-1][feat], 0.75, featProbPlusList[ind-1][feat][self.reduceGram(gram)])
						else:
							featProbPlusList[ind][feat][gram] = self.absoluteDiscount(trainObj.allFeaturePlus[ind-1][feat][gram], overallFeatPlus[ind-1][feat], 0.75, featProbPlusList[ind-1][gram])
			for feat in self.allFeatureMinus[ind-1]:
				for gram in self.allGramSet[ind-1]:
					if feat not in featProbMinusList[ind]:
						featProbMinusList[ind][feat] = defaultdict(int)
					if gram not in self.allFeatureMinus[ind-1][feat]:
						if ind>1:
							featProbMinusList[ind][feat][gram] = self.absoluteDiscount(0, overallFeatMinus[ind-1][feat], 0.75, featProbMinusList[ind-1][feat][self.reduceGram(gram)])
						else:
							featProbMinusList[ind][feat][gram] = self.absoluteDiscount(0, overallFeatMinus[ind-1][feat], 0.75, featProbMinusList[ind-1][gram])

					else:
						if ind>1:
							featProbMinusList[ind][feat][gram] = self.absoluteDiscount(trainObj.allFeatureMinus[ind-1][feat][gram], overallFeatMinus[ind-1][feat], 0.75, featProbMinusList[ind-1][feat][self.reduceGram(gram)])
						else:
							featProbMinusList[ind][feat][gram] = self.absoluteDiscount(trainObj.allFeatureMinus[ind-1][feat][gram], overallFeatMinus[ind-1][feat], 0.75, featProbMinusList[ind-1][gram])

		priorPlus={}
		priorMinus={}
		for feat in overallFeatPlus[4]:
			priorPlus[feat] = overallFeatPlus[4][feat] / (overallFeatPlus[4][feat] + overallFeatMinus[4][feat])

		for feat in overallFeatMinus[4]:
			priorMinus[feat] = overallFeatMinus[4][feat] / (overallFeatPlus[4][feat] + overallFeatMinus[4][feat])
		return((priorPlus, priorMinus, featProbPlusList,featProbMinusList))

	def evalDevData(self,trainObj):
		priorPlus, priorMinus, featProbPlusList,featProbMinusList=trainObj.calcAbsoluteDiscount(trainObj)
		devPlus = defaultdict(dict)
		devMinus = defaultdict(dict)
		proposedFeatures = defaultdict(list)
		matched = defaultdict(int)
		proposed = defaultdict(int)
		actual = defaultdict(int)
		correct = 0

		totalErrors = 0
		totalInTrain = 0
		totalUnkWords = 0
		inTrain = 0
		unkWords = 0
		flag = True


		for rows in self.wordTag:
			word = rows[1]
			flag = True
			if word not in trainObj.wordSet:
				totalUnkWords += 1
				continue
			else:
				totalInTrain += 1

			# token += 1
			wgrams = self.getnGrams(word,5)
			tags = self.wordTag[rows]
			features = tags.split(';')
			for feat in features:
				actual[feat]+=1
				for gram in wgrams:
					if word not in devPlus:
						devPlus[feat]=defaultdict(int)
					if word not in devMinus:
						devMinus[feat]=defaultdict(int)

					if word not in devPlus[feat]:
						devPlus[feat][word] = featProbPlusList[4][feat][gram]
					devPlus[feat][word] *= featProbPlusList[4][feat][gram]

					if word not in devMinus[feat]:
						devMinus[feat][word] = featProbMinusList[4][feat][gram]
					devMinus[feat][word] *= featProbMinusList[4][feat][gram]

				if feat not in self.featurePlus:
					self.featurePlus[feat]=defaultdict(int)
				self.featurePlus[feat][word] = priorPlus[feat] * devPlus[feat][word]
				if feat not in self.featureMinus:
					self.featureMinus[feat]=defaultdict(int)
				self.featureMinus[feat][word] = priorMinus[feat] * devMinus[feat][word]


				# if word not in proposedFeatures:
				# 	proposedFeatures[word]=[]
				if self.featurePlus[feat][word] > self.featureMinus[feat][word]:
					if feat not in proposedFeatures[word]:
						proposedFeatures[word].append(feat)

			# store the counts for proposed
			for feat in proposedFeatures[word]:
				proposed[feat]+=1
				if feat in features:
					matched[feat]+=1
				for fet in features:
					if feat not in features or fet not in proposedFeatures[word]:
						if flag == True:
							if word in trainObj.wordSet:
								inTrain += 1
							else:
								unkWords += 1
							flag = False

			if set(features)== set(proposedFeatures[word]):
				correct += 1
			else:
				totalErrors += 1

		print("Total:	"+str(len(self.wordTag)))
		print("Total UNK words:	  " +str(totalUnkWords))
		print("Total Errors:	"+str(totalErrors))
		print("Correct:	"+str(correct))
		print("Accuracy:	"+str(correct/len(self.wordTag)))
		return((actual, matched, proposed, set(self.wordTag.values())))



class RawFGrams:
	def __init__(self,file):
		self.featSet=set()
		self.gramSet=set()
		self.wordTag={}
		self.wordSet=set()
		self.featurePlus={}
		self.featureMinus={}

		with open(file) as fin:
			lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
		for nr,rows in enumerate(lines):
			word=str(rows[1])
			self.wordSet.add(word)
			tagDim=str(rows[5])
			tags=tagDim.split(';')
			self.wordTag[(nr,word)]=tagDim
			for t in tags:
				self.featSet.add(t)

	def getnGrams(self,word,n):
		nullList=[]
		for null in range(1,n):
			nullList.append('**')
		padded = nullList+list(word)+ ['**']
		ngramlist = []
		for i in range(4, len(word)+5):
			ngramlist.append(tuple(padded[i-4 : i+1]))
		return(ngramlist)

	def getFeatureSets(self):
		for sentInd,word in self.wordTag:
			tags=self.wordTag[(sentInd,word)].split(';')
			wgrams=self.getnGrams(word,5)
			for feature in self.featSet:
				for wg in wgrams:
					self.gramSet.add(wg)
					if feature in tags:
						if feature not in self.featurePlus:
							self.featurePlus[feature]=defaultdict(int)
						self.featurePlus[feature][wg]+=1
					else:
						if feature not in self.featureMinus:
							self.featureMinus[feature]=defaultdict(int)
						self.featureMinus[feature][wg]+=1

	def evalData(self):
		overallFeatPlus=defaultdict(int)
		overallFeatMinus=defaultdict(int)
		for feat in self.featurePlus:
			for wg in self.featurePlus[feat]:
				if feat not in overallFeatPlus:
					overallFeatPlus[feat]=0
				overallFeatPlus[feat] += self.featurePlus[feat][wg]

		for feat in self.featureMinus:
			for wg in self.featureMinus[feat]:
				if feat not in overallFeatMinus:
					overallFeatMinus[feat]=0
				overallFeatMinus[feat] += self.featureMinus[feat][wg]
		return((overallFeatPlus,overallFeatMinus))

	def calcPrior(self):
		overallFeatPlus,overallFeatMinus=self.evalData()
		priorPlus = {}
		priorMinus = {}

		for feat in overallFeatPlus:
			priorPlus[feat] = overallFeatPlus[feat] / (overallFeatPlus[feat] + overallFeatMinus[feat])

		for feat in overallFeatMinus:
			priorMinus[feat] = overallFeatMinus[feat] / (overallFeatPlus[feat] + overallFeatMinus[feat])

		return((priorPlus,priorMinus))

	def addOneSmoothing(self):
		gPlus={}
		gMinus={}
		overallFeatPlus,overallFeatMinus=self.evalData()
		for feat in self.featurePlus:
			for gram in self.gramSet:
				if feat not in gPlus:
					gPlus[feat] = {}
				if gram not in self.featurePlus[feat]:
					gPlus[feat][gram] = 1 / (overallFeatPlus[feat] + len(self.gramSet))
				else:
					gPlus[feat][gram] = (self.featurePlus[feat][gram] + 1) / (overallFeatPlus[feat] + len(self.gramSet))

		for feat in self.featureMinus:
			for gram in self.gramSet:
				if feat not in gMinus:
					gMinus[feat] = {}
				if gram not in self.featureMinus[feat]:
					gMinus[feat][gram] = 1 / (overallFeatMinus[feat] + len(self.gramSet))
				else:
					gMinus[feat][gram] = (self.featureMinus[feat][gram] + 1) / (overallFeatMinus[feat] + len(self.gramSet))
		return((gPlus,gMinus))

	def evalDevData(self,trainObj):
		trainObj.getFeatureSets()
		priorPlus,priorMinus=trainObj.calcPrior()
		gPLus,gMinus=trainObj.addOneSmoothing()
		devPlus = defaultdict(dict)
		devMinus = defaultdict(dict)
		proposedFeatures = defaultdict(list)
		matched = defaultdict(int)
		proposed = defaultdict(int)
		actual = defaultdict(int)
		correct = 0

		totalErrors = 0
		totalInTrain = 0
		totalUnkWords = 0
		inTrain = 0
		unkWords = 0
		flag = True


		for rows in self.wordTag:
			word = rows[1]


			flag = True
			if word not in trainObj.wordSet:
				totalUnkWords += 1
				continue
			else:
				totalInTrain += 1

			# token += 1
			wgrams = self.getnGrams(word,5)
			tags = self.wordTag[rows]
			features = tags.split(';')
			for feat in features:
				actual[feat]+=1
				for gram in wgrams:
					if word not in devPlus:
						devPlus[feat]=defaultdict(int)
					if word not in devMinus:
						devMinus[feat]=defaultdict(int)

					if word not in devPlus[feat]:
						devPlus[feat][word] = gPLus[feat][gram]
					devPlus[feat][word] *= gPLus[feat][gram]

					if word not in devMinus[feat]:
						devMinus[feat][word] = gMinus[feat][gram]
					devMinus[feat][word] *= gMinus[feat][gram]

				if feat not in self.featurePlus:
					self.featurePlus[feat]=defaultdict(int)
				self.featurePlus[feat][word] = priorPlus[feat] * devPlus[feat][word]
				if feat not in self.featureMinus:
					self.featureMinus[feat]=defaultdict(int)
				self.featureMinus[feat][word] = priorMinus[feat] * devMinus[feat][word]


				# if word not in proposedFeatures:
				# 	proposedFeatures[word]=[]
				if self.featurePlus[feat][word] > self.featureMinus[feat][word]:
					if feat not in proposedFeatures[word]:
						proposedFeatures[word].append(feat)

			# store the counts for proposed
			for feat in proposedFeatures[word]:
				proposed[feat]+=1
				if feat in features:
					matched[feat]+=1
				for fet in features:
					if feat not in features or fet not in proposedFeatures[word]:
						if flag == True:
							if word in trainObj.wordSet:
								inTrain += 1
							else:
								unkWords += 1
							flag = False

			if set(features)== set(proposedFeatures[word]):
				correct += 1
			else:
				totalErrors += 1
		print("Total:	"+str(len(self.wordTag)))
		print("Total UNK words:  " +str(totalUnkWords))
		print("Total Errors:	"+str(totalErrors))
		print("Correct:	"+str(correct))
		print("Total errors:  "+str(totalErrors))
		print("Accuracy:	"+str(correct/len(self.wordTag)))
		return((actual, matched, proposed, set(self.wordTag.values())))

class BaseLine:
	def __init__(self,name):
		self.name=name
		print("Language:  "+name)

	def getFrequent(self,file):
		vocab=defaultdict(dict)
		mostFreq=defaultdict(str)
		with open(file) as fin:
			lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
		for rows in lines:
			word=str(rows[1])
			tagDim=str(rows[5])
			if tagDim in vocab[word]:
				vocab[word][tagDim]+=1
			else:
				vocab[word][tagDim]=1
		for word in vocab:
			mostFreq[word]=max(vocab[word],key=vocab[word].get)
		return(mostFreq)

	def evalDevData(self,testFile,vocab):
		correct=0
		testOnly=0
		matchedTags=defaultdict(int)
		proposedTags=defaultdict(int)
		actualTags=defaultdict(int)
		tagSet=set()
		with open(testFile) as fin:
			lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
		for rows in lines:
			word=str(rows[1])
			tagDim=str(rows[5])
			actualTags[tagDim]+=1
			tagSet.add(tagDim)
			if word in vocab:
				trainT=vocab[word]
				proposedTags[tagDim]+=1
				if tagDim == trainT:
					correct+=1
					matchedTags[tagDim]+=1
			else:
				testOnly+=1

		print("Total:  "+str(len(lines)))
		print("Total UNK words:  "+str(testOnly))
		print("Correct:  "+str(correct))
		print("Over all accuracy:  "+str(correct/(len(lines))))
		return((actualTags,matchedTags,proposedTags,tagSet))

def printEvalMatrics(actual,matched,proposed,tags,bl=False):
	f1 = lambda P,R : 2*(P*R)/(P+R)
	matchedCount=sum(matched.values())
	proposedCount=sum(proposed.values())
	actualCount=sum(actual.values())
	precision=matchedCount/proposedCount
	recall=matchedCount/actualCount
	print("Over all precision:  "+str(precision))
	print("Over all recall:  "+str(recall))
	print("Overall F1-score:  "+str(f1(precision,recall)))
	print()
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
			print("Tag  "+t)
			print("Precision:  " + str(P))
			print("Recall:  " + str(R))
			print("F1-score:  " + str(f1(P, R)))
			print("___________________________________")
	print()


def main():
	trainFiles=['./Data/en_ewt-um-train.conllu','./Data/es_ancora-um-train.conllu','./Data/ru_gsd-um-train.conllu','./Data/tr_imst-um-train.conllu','./Data/zh_gsd-um-train.conllu','./Data/sa_ufal-um-train.conllu']
	testFiles=['./Data/en_ewt-um-dev.conllu','./Data/es_ancora-um-dev.conllu','./Data/ru_gsd-um-dev.conllu','./Data/tr_imst-um-dev.conllu','./Data/zh_gsd-um-dev.conllu','./Data/sa_ufal-um-dev.conllu']
	lang=['English', 'Spanish', 'Russian', 'Turkish', 'Chinese', 'Sanskrit']
	for i in range(len(lang)):
		#Baseline
		print("Running base line on "+lang[i])
		train=BaseLine(lang[i])
		vocab=train.getFrequent(trainFiles[i])
		actual,matched,proposed,tags=train.evalDevData(testFiles[i],vocab)
		printEvalMatrics(actual,matched,proposed,tags)
		#Raw 5 grams
		print("Running Raw 5 gram classifier on "+lang[i])
		train=RawFGrams(trainFiles[i])
		test=RawFGrams(testFiles[i])
		actual,matched,proposed,tags=test.evalDevData(train)
		printEvalMatrics(actual,matched,proposed,tags)
		# Absolute Discounting
		print("Running Interpolated 5 gram classifier on "+lang[i])
		train=Interpolated(trainFiles[i])
		test=Interpolated(testFiles[i])
		actual,matched,proposed,tags=test.evalDevData(train)
		printEvalMatrics(actual,matched,proposed,tags)
		# if i<len(lang)-1:
		# 	response=input("Run on "+lang[i+1]+"? y/n")
		# 	if response=='n' and response=='N':
		# 		exit()


if __name__ == "__main__":
	main()
