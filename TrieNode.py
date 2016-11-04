import numpy as np
import json

class TrieNode:
	def __init__(self,classes):
		self.child={}
		self.is_word=False
		self.class_frequencies=np.zeros((1,classes)).astype(int)

	def add_child(self,a,which_class,classes):
		
		if not self.child.has_key(a[0]):
			self.child[a[0]]=TrieNode(classes)

		if len(a)>1:
			return self.child[a[0]].add_child(a[1:],which_class,classes)
		else:
			toReturn=False
			if self.child[a[0]].is_word is False:
				toReturn=True
			self.child[a[0]].is_word=True
			self.child[a[0]].class_frequencies[0,which_class]=self.child[a[0]].class_frequencies[0,which_class]+1
			return toReturn

	def check_word_and_probability(self,a):
		if a[0] not in self.child:
			return (False,None)

		if(len(a)==1):
			if self.child[a[0]].is_word:
				return (True,self.child[a[0]].class_frequencies)
			else:
				return (False,None)
		else:
			return self.child[a[0]].check_word_and_probability(a[1:])

