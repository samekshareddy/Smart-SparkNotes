# imports needed and logging
from summarizer import Summarizer
import time
import json

class SmartNote:


	# doc_name = 'sample_document.json'
	num_topics = 5 
	itertations = 100
	# title = 'Applying building information modeling to integrate schedule and cost for establishing construction progress curves'
	thematic_threshold = 5
	# desired_sen = 5

	def getSummary(self,paper,title,num_sentences = 5, query = ""):
		
		s = Summarizer(paper, self.num_topics, self.itertations, title, self.thematic_threshold, num_sentences)
		
		if query == "":
			return s.retrieve_summary()
		
		else:
			s.retrieve_summary()
			return s.retrieve_query_summary(query)



