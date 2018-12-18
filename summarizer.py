import nltk
from collections import defaultdict
from textblob import TextBlob
from nltk.tag import pos_tag
import json
import time
import numpy as np
import math
import gensim 

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

from nltk.corpus import stopwords
from score import ScoreGenerator
from topics import Topics



class Summarizer(object):
    '''
    doc_name - file name to retrieve document file
    num_topics - number of topics for topic model
    itertations - number of iteration for topic model
    title - title of paper for score generator
    thematic_threshold - threhold for thematic words for score generator
    '''

    def __init__(self, paper, num_topics, itertations, title, thematic_threshold, desired_summary_sen):
        self.num_topics = num_topics

        # with open(doc_name) as f:
        #     paper = json.load(f)

        # initialize topic model
        self.t = Topics(paper, num_topics, itertations)
        
        # 
        self.t.get_model()
        # the topics distribution is saved in t.topics_dict

        # self.t.get_coverage()

        # initialize score generator

        ############### future plan is to save this permanently
        # model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True, )

        # word_vectors = model.wv

        # fname = get_tmpfile("vectors.kv")
        # word_vectors.save(fname)
        ############### future plan is to save this permanently

        keyword_dict = defaultdict(int)
        for topic in self.t.topics_dict:
            for word in self.t.topics_dict[topic]:
                keyword_dict[word] += self.t.topics_dict[topic][word] 

        keywords=list(keyword_dict.keys())

        score_weights =[0.5,0.1,0.2,0.285*4,0.1,0.0277*4,0.1, 0.9]

        keywords_weights=list(keyword_dict.values())

        self.sg = ScoreGenerator(title.lower(), paper, thematic_threshold, keywords, keywords_weights, score_weights)

        self.desired_summary_sen = desired_summary_sen
        
    # function to retireve summary of a paper based on weights
    def retrieve_summary(self):

        result = self.sg.calculate()

        total_scores = np.array(result[1])

        rankings = []
        for sentence_scores in total_scores:
            sentence_scores = np.array(sentence_scores)
            sentence_ranking = np.argsort(sentence_scores)
            rankings.append(sentence_ranking)

        desired_summary_sen = self.desired_summary_sen

        #display the result of the best n sentence from current paragraph as candidate summary
        doc_keys = list(self.sg.document.keys())
        candidate_summary = ''

        self.ns = [] 
        num_sections = len(self.sg.document) 
        # maximum_sentences_per_section - msps
        msps = 0
        for item in self.sg.document:
            msps += len(self.sg.document[item].sentences)  

        for item in self.sg.document:
            self.ns.append(len(self.sg.document[item].sentences) / msps)
            
        self.ns = np.array(self.ns) * desired_summary_sen

        candidate_summary = '...'

        sections = ['abstract', 'intro', 'conclusion']

        section_key = defaultdict()
        section_idx = defaultdict()

        for key_idx in range(len(doc_keys)):
            key = doc_keys[key_idx]
            for section in sections:
                if section in key.lower():
                    section_key[section] = key
                    section_idx[section] = key_idx

        sen_num = 0
        while (sen_num < self.desired_summary_sen):

            for section in section_key:
                if section != 'abstract':
                    section_id = section_idx[section]
                    sentence_ranking = rankings[section_id]
                    n = self.ns[section_id]
                    n = math.ceil(n)

                    if (sen_num + n) > self.desired_summary_sen:
                        n = max(0, min(n, self.desired_summary_sen - sen_num))

                    if n != 0:
                        d = self.sg.document[doc_keys[section_id]]
                        display_idxs = np.sort(sentence_ranking[-n:])

                        for sen_idx in display_idxs:
                            sen = self.sg.document[doc_keys[section_id]].sentences[sen_idx]
                            s = str(sen)
                            s = s.split(' ', 1)
                            word1 = s[0].capitalize()
                            rest_sen = ""
                            if len(s) > 1:
                                rest_sen = s[1]


                            sen = word1 + " " +rest_sen
                            candidate_summary += str(sen)+ ' '
                            sen_num += n
            break

        while (sen_num < self.desired_summary_sen):
            for rank_index in range(len(rankings)):
                sentence_ranking = rankings[rank_index]
                n = self.ns[rank_index]
                n = math.ceil(n)
                if (sen_num + n) > self.desired_summary_sen:
                    n =max(0, min(n, self.desired_summary_sen - sen_num))

                if n != 0 and doc_keys[rank_index] not in list(section_idx.values()):
                
                    d = self.sg.document[doc_keys[rank_index]]
                    display_idxs = np.sort(sentence_ranking[-n:])

                    for sen_idx in display_idxs:
                        sen = self.sg.document[doc_keys[rank_index]].sentences[sen_idx]
                        s = str(sen)
                        s = s.split(' ', 1)
                        word1 = s[0].capitalize()
                        rest_sen = ""
                        if len(s) > 1:
                            rest_sen = s[1]

                        sen = word1 + " " +  rest_sen
                        candidate_summary += str(sen)+ ' '
                        sen_num += n
            break

        return candidate_summary

    def retrieve_query_summary(self, query):
        doc_keys = list(self.sg.document.keys())
        
        query_score = self.sg.query(query)
        query_total_scores = []

        #initialize weight for user query
        user_query_weight = 4

        for item_idx in range(len(query_score)):
            q_sentence_score = np.array(query_score[item_idx])
            sentence_scores = np.array(self.sg.total_scores[item_idx])
            query_total_scores.append(user_query_weight*q_sentence_score + sentence_scores)
            
        query_rankings = []
        for sentence_scores in query_total_scores:
            sentence_ranking = np.argsort(sentence_scores)
            query_rankings.append(sentence_ranking) 
            
        query_candidate_summary = '...'

        sections = ['abstract', 'intro', 'conclusion']

        section_key = defaultdict()
        section_idx = defaultdict()

        for key_idx in range(len(doc_keys)):
            key = doc_keys[key_idx]
            for section in sections:
                if section in key.lower():
                    section_key[section] = key
                    section_idx[section] = key_idx

        sen_num = 0
        while (sen_num < self.desired_summary_sen):

            for section in section_key:
                if section != 'abstract':
                    section_id = section_idx[section]
                    sentence_ranking = query_rankings[section_id]
                    n = self.ns[section_id]
                    n = math.ceil(n)

                    if (sen_num + n) > self.desired_summary_sen:
                        n = max(0, min(n, self.desired_summary_sen - sen_num))

                    if n != 0:
                        d = self.sg.document[doc_keys[section_id]]
                        display_idxs = np.sort(sentence_ranking[-n:])

                        for sen_idx in display_idxs:
                            sen = self.sg.document[doc_keys[section_id]].sentences[sen_idx]
                            s = str(sen)
                            s = s.split(' ', 1)
                            word1 = s[0].capitalize()
                            rest_sen = ""
                            if len(s) > 1:
                                rest_sen = s[1]


                            sen = word1 + " " + rest_sen
                            query_candidate_summary += str(sen)+ ' '
                            sen_num += n
            break

        while (sen_num < self.desired_summary_sen):
            for rank_index in range(len(query_rankings)):
                sentence_ranking = query_rankings[rank_index]
                n = self.ns[rank_index]
                n = math.ceil(n)
                if (sen_num + n) > self.desired_summary_sen:
                    n =max(0, min(n, self.desired_summary_sen - sen_num))

                if n != 0 and doc_keys[rank_index] not in list(section_idx.values()):
                
                    d = self.sg.document[doc_keys[rank_index]]
                    display_idxs = np.sort(sentence_ranking[-n:])

                    for sen_idx in display_idxs:
                        sen = self.sg.document[doc_keys[rank_index]].sentences[sen_idx]
                        s = str(sen)
                        s = s.split(' ', 1)
                        word1 = s[0].capitalize()
                        rest_sen = ""
                        if len(s) > 1:
                            rest_sen = s[1]

                        sen = word1 + " " + rest_sen
                        query_candidate_summary += str(sen)+ ' '
                        sen_num += n
            break


        return query_candidate_summary