import spacy

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('stopwords')

import gensim
from gensim import corpora
from gensim.models.ldamodel	import LdaModel

import time
import numpy as np
import os
import json
# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--d", type=str, help="Document to Parse")
# parser.add_argument("--k", default= 5, type=int, help="Number of Topics")
# parser.add_argument("--it", default= 100, type=int, help="Number of LDA Iterations")

class Topics:
	def __init__(self, data, num_topics, itertations):
		'''
		initilizes the a Topics instance and tokenizes the input document
		'''
		# self.text = open(filename, 'r').read()
		# self.filename = filename[0:-4]

		self.wholeDocument = data

		# if not os.path.exists(self.output_directory):
		# 	os.makedirs(self.output_directory)

		# self.filename = filename
		if isinstance(data, dict):
			self.Document = data
			self.text = ''
			for key in data:
				self.text += data[key] + ' '
		
		else:
			self.text = data

		self.languageModel = spacy.load('en')
		self.lemmatizer = WordNetLemmatizer()
		self.en_stop = set(nltk.corpus.stopwords.words('english')).union(set(['the', 'The']))
		self.num_topics	= num_topics
		self.itertations = itertations

		self.tokenize()
		self.vocab_size = len(self.document)


	def tokenize(self):
		'''
		function that returns all the sentences in the document after:
			1- lemmatizing every word
			2- removing stop words
			3- removing short words
		'''
		self.document = self.languageModel(self.text)
		self.sentences = []
		# counter = 0
		for sent in self.document.sents:
			sentence = [self.lemmatizer.lemmatize(word.lower()) for word in sent.text.split()
							if (word.lower() not in self.en_stop and len(word)>=3)] 
			self.sentences.append(sentence)
			# counter+=1
		# print(counter)
		# print(len(self.sentences))


	def tokenize_testDoc(self, testDoc):
		'''
		function that returns all the sentences in a test document in the same way as tokenize.
		'''
		# text = open(testDoc, 'r').read()
		text = testDoc
		document = self.languageModel(text)
		sentences = []
		for sent in document.sents:
			sentence = [self.lemmatizer.lemmatize(word.lower()) for word in sent.text.split()
							if word.lower() not in self.en_stop	and len(word)>3] 
			sentences.append(sentence)
		return sentences


	def get_model(self):
		'''
		function that builds a LDA model for the whole document being summarized
		the function also saved the model based on the document's name
		'''
		dictionary = corpora.Dictionary(self.sentences)
		corpus = [dictionary.doc2bow(sentence) for sentence in self.sentences]
		self.ldamodel = LdaModel(corpus, num_topics = self.num_topics, 
							id2word=dictionary, passes=self.itertations, random_state = 0)

		self.print_model()

	def print_model(self):
		'''
		function that prints topic distributions over all words in the original document
		and saves them into a json file
		'''
		topics = self.ldamodel.print_topics(num_words=self.vocab_size)

		self.topics_dict = {}
		for topic in topics:
			topic_nb = topic[0]
			distribution = topic[1].split('+')
			topic_dict = {}
			for word_prob in distribution:
				word_prob_ = word_prob.split('*')
				topic_dict[word_prob_[1][1:-2]] = float(word_prob_[0])

			self.topics_dict[topic_nb] = topic_dict


		# with open(self.output_directory + self.filename + '_topics_dict.json', 'w') as output:
		# 	json.dump(self.topics_dict, output)

		# for key in self.topics_dict:
		# 	print("Topic ", key, " Distribution: ", self.topics_dict[key], '\n')


	def save_model(self):
		'''
		functions that saves an LDA model
		'''
		self.ldamodel.save(self.output_directory + self.filename + '_ldaModel.gensim')

	def load_model(self):
		'''
		functions that loads an already existing LDA model
		'''
		self.ldamodel = LdaModel.load(self.output_directory + self.filename + '_ldaModel.gensim')

	def get_topic_dist(self, doc):
		'''
		funciton that can be used to get the topic distribution for every paragraph
	 	of the original document
		'''
		paragraph = self.tokenize_testDoc(doc)
		# print(len(paragraph))
		'''
		reuse the dictionay built based on the original document
		for our case this dict should cover all the the paragraphs
		'''
		dictionary = corpora.Dictionary(self.sentences) 
		corpus = [dictionary.doc2bow(sentence) for sentence in paragraph]
		dists = []
		for i, cor in enumerate(corpus):
			dist = self.ldamodel[cor]
			dists.append(dist)

		# print(dists)
		return dists

	def get_coverage(self):
		'''
		this function finds how much each topic is covered in every paragraph of the document
		'''
		self.coverage = {}
		for paragraph in list(self.Document.keys()):
			if len(self.Document[paragraph]) != 0:
				self.coverage[paragraph] = self.get_paragraph_coverage(paragraph)
			else:
				self.coverage[paragraph] = np.zeros([self.num_topics]).tolist()
		
		# self.print_coverage()

		# with open(self.output_directory + self.filename + '_paragraph_coverage_dict.json', 'w') as output:
		# 	json.dump(self.coverage, output)

		return self.coverage


	def get_paragraph_coverage(self, paragraph_key):
		'''
		this function finds topic coverage per paragraph 
		by averaging over topic coverage per sentence

		'''
		paragraph = self.Document[paragraph_key]
		dists = self.get_topic_dist(paragraph)
		np_dist = []
		for dist in dists:
			np_sent_dist = []
			if len(dist) == self.num_topics:
				for tup in dist:
					np_sent_dist.append(tup[1])
				np_dist.append(np_sent_dist)
			else:
				np_sent_dist=np.zeros([self.num_topics])
				for tup in dist:
					np_sent_dist[int(tup[0])] = tup[1]
				np_dist.append(np_sent_dist.tolist())

		paragraph_coverage = np.average(np.array(np_dist), axis=0)
		return paragraph_coverage.tolist()

	
	# def print_coverage(self):
	# 	for key in self.coverage:
	# 		print('\n', key, ': ', self.coverage[key])


if __name__ == '__main__':
	# t = Topics('data.txt', 3, 50)
	# filename = parser.parse_args().d
	# num_topics = parser.parse_args().k
	# itertations = parser.parse_args().it

	# with open(filename +'.json', 'r', encoding='utf8') as outfile:
	# 	data = json.load(outfile)

	# data needs to be the head/content from Sameksha
	num_topics = 5
	itertations = 100


	# start = time.time()
	t = Topics(data, num_topics, itertations)
	
	t.get_model()
	# the topics distribution is saved in t.topics_dict

	t.get_coverage()
	# the topics paragraph coverage is saved in t.coverage

	# that should be all we need from this part
	
	print('\n Elapsed Time: ', round(time.time() - start, 3), ' s')
	


