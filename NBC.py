from TrieNode import TrieNode
import numpy as np
from nltk import stem
import json
import pickle
import csv
class NBC:
	def __init__(self,class_list=None):
		self.stemmer=stem.PorterStemmer()
		self.vocabulary=0
		if class_list is not None:
			self.set_class_list(class_list)

	def set_class_list(self,class_list):
		self.class_list=class_list
		self.classes=len(class_list)
		self.class_frequency=np.zeros((1,self.classes))
		self.class_word_count=np.zeros((1,self.classes))
		self.root=TrieNode(self.classes)

	def stringToClass(self,classInString):		
		for i in range(0,self.classes):
			if self.class_list[i] is classInString:
				return i

	def classToString(self,class_number):
		return self.class_list[class_number]

	def train(self,sentences,classes):
		for i in range(0,len(sentences)):
			self.class_frequency[0,classes[i]]+=1
			for word in sentences[i].lower().split(" "):
				self.class_word_count[0][classes[i]]+=1
				if(self.root.add_child(self.stemmer.stem(word),classes[i],self.classes)):
					self.vocabulary+=1

	def test(self,sentence):
		result=self.class_frequency/np.sum(self.class_frequency)
		for word in sentence.lower().split(" "):
			is_word,class_frequency=self.root.check_word_and_probability(self.stemmer.stem(word))
			if is_word:
				result+=np.log((class_frequency+1)/(self.class_word_count+self.vocabulary))
			else:
				result+=np.log((np.zeros((1,self.classes))+1)/(self.class_word_count+self.vocabulary))
		return np.argmax(result)

	#Load Trained Model
	@staticmethod
	def loadModel():
		with open('model.json','r') as file:
			return pickle.load(file)
	
	#Save Trained Model 	
	def saveModel(self):
		with open('model.json', 'w') as outfile:
			pickle.dump(self,outfile)
