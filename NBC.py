from TrieNode import TrieNode
import numpy as np
from nltk import stem
import json
import pickle
class NBC:
	def __init__(self,classes=None):
		self.class_frequency=np.zeros((1,classes))
		self.vocabulary=0
		self.class_word_count=np.zeros((1,classes))
		self.stemmer=stem.PorterStemmer()
		self.classes=classes
		self.root=TrieNode(classes)
		self.trainedOn=0

	def train(self,sentences,classes):
		self.trainedOn+=len(classes)
		for i in range(0,len(sentences)):
			self.class_frequency[0,classes[i]]+=1
			for word in sentences[i].lower().split(" "):
				self.class_word_count[0][classes[i]]+=1
				if(self.root.add_child(self.stemmer.stem(word),classes[i],self.classes)):
					self.vocabulary+=1

	def test(self,sentences):
		output=[]
		for i in range(0,len(sentences)):
			result=self.class_frequency/self.trainedOn
			for word in sentences[i].lower().split(" "):
				is_word,class_frequency=self.root.check_word_and_probability(self.stemmer.stem(word))
				if is_word:
					result+=np.log((class_frequency+1)/(self.class_word_count+self.vocabulary))
			output.append(np.argmax(result))
		return output

	@staticmethod
	def loadModel():
		with open('model.json','r') as file:
			return pickle.load(file)
		
	def saveModel(self):
		with open('model.json', 'w') as outfile:
			pickle.dump(self,outfile)