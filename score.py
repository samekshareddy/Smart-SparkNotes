import nltk
from collections import defaultdict
from textblob import TextBlob
import gensim 
import numpy as np
import pdb
import string

from nltk.tag import pos_tag

from nltk.corpus import stopwords

from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

class ScoreGenerator(object):
    """
    A sentence score generator that rates setences based on different evaluation metrics
    """

    def __init__(self, title, document, thematic_threshold, keywords=None, keywords_weights=None, score_weights=None):
        """
        Return Score Generator based on a document.
        Parameter: 
        keywords - a string, currently assumed to be the title of the document
        document - TextBlob, think of it as high powered python string, in real life this is 
        assumed to be the paragraph
        """
        # title for title feature score calculation
        self.stopwords =set(stopwords.words('english'))

        self.title = []
        for word in title:
            if word not in self.stopwords:
                self.title.append(word.lower())

        # key words to be used for weight score
        if keywords:
            self.keywords = keywords
        else:
            self.keywords = self.title

        # weight of each key word 
        if keywords_weights:
            self.keywords_weights = np.array(keywords_weights)
        else:
            self.keywords_weights = np.array([1/len(self.keywords)] * len(title.split()))

        self.thematic_threshold = thematic_threshold

        # document to save 
        # turn document string into lower case for gensim
        self.document = defaultdict()
        self.longest_sentence = defaultdict()
        self.thematic_words = defaultdict()
        self.total_thematic_words = defaultdict()

        for item in document:
            self.document[item] = TextBlob(document[item].lower())
            self.longest_sentence[item] = self._find_longest_sentence(item)
            self.thematic_words[item], self.total_thematic_words[item] = self._find_thematic_words(self.thematic_threshold, item)

        # weight of each score
        if score_weights: 
            self.score_weights = score_weights
        else:
            self.score_weights = [1/8] * 8

        # load model
        # model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True, )

        # save word_vecs
        fname = get_tmpfile("vectors.kv")
        # word_vectors = KeyedVectors.load(fname, mmap='r')
        self.word_vectors = KeyedVectors.load(fname, mmap='r')
        # calculate similarity for all words encountered
        self.word_sim = self._calculate_word_similairty()

    def _calculate_word_similairty(self):
        # initialize a table to remove punctuation from string
        table = str.maketrans(dict.fromkeys(string.punctuation))
        # get all unique word from document
        doc_words = defaultdict()

        # add words from document
        for item in self.document:
            text = self.document[item]
            words = text.words
            for word in words:
                if word not in doc_words and word in self.word_vectors.vocab:
                    doc_words[word] = self.word_vectors[word]

        # add words from title and keywords
        for word in self.title: 
            if word not in doc_words and word in self.word_vectors:
                doc_words[word] = self.word_vectors[word]

        for word in self.keywords:
            if word not in doc_words and word in self.word_vectors:
                doc_words[word] = self.word_vectors[word]

        self.word_to_id = defaultdict()

        words = list(doc_words.keys())
        for word_id in range(len(words)):
            word = words[word_id]
            self.word_to_id[word] = word_id

        word_vecs = []
        for word in doc_words:
            word_vecs.append(doc_words[word])
            
        word_vecs = np.array(word_vecs)

        # save a copy of the word_vec for query words calculation
        self.relevant_word_vector = word_vecs

        cos_sim_nominator = np.dot(word_vecs, np.transpose(word_vecs))

        word_vec_mags = np.linalg.norm(word_vecs, axis=1)

        sim = np.transpose(cos_sim_nominator / word_vec_mags) / word_vec_mags 

        return sim

    def _is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
     
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
     
        return False
        
    def _find_longest_sentence(self, item):
        '''
        Find longest sentence in a document
        '''

        max_length = 0
        longest_sentence = ''
        
        for sentence in self.document[item].sentences:
            if len(sentence.words) > max_length:
                max_length = len(sentence.words)
                longest_sentence = sentence
        
        return longest_sentence
        
    def _find_thematic_words(self, thematic_threshold, item):
        '''
        Return list of thematic words and number of thematic words
        Parameter: 
        thematic_threshold - amount of time a word need to appear to be counted
        item - the paragraph in the paper
        '''
        thematic_words = []
        total_thematic_words = 0

        word_set = set(self.document[item].words)

        for word in word_set:
            if self.document[item].words.count(word) > thematic_threshold:
                if word not in self.stopwords:
                    thematic_words.append(word)
                    total_thematic_words += self.document[item].words.count(word)

        return thematic_words, total_thematic_words

    def _title_feature(self, item, i):
        '''
        Return title feature score
        score = |Intersection (keywords of sentence, keywords of document)| 
                / |Union (keywords of sentence, keywords of document)|

        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''

        sentence = self.document[item].sentences[i]
        intersection_total = 0
        for keyword in self.title:
            if sentence.words.count(keyword) > 0: 
                intersection_total += sentence.words.count(keyword)
            else:
                for word in sentence.words:
                    if word in self.word_to_id and keyword in self.word_to_id:
                        keyword_id = self.word_to_id[keyword]
                        word_id = self.word_to_id[word]
                        intersection_total += self.word_sim[keyword_id][word_id]
            


        return intersection_total / (len(self.title)+len(sentence.words))

    def _sentence_length(self, item, i):
        """
        Return sentence length score
        score = sentence_length / longest_sentence_length

        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        """
        sentence = self.document[item].sentences[i]

        return len(sentence.words) / len(self.longest_sentence[item].words)

    def _sentence_position(self, item, i):
        '''
        Return sentence position score
        score = (total_sentence_num - i) / total_sentence_num

        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''
        sentence = self.document[item].sentences[i]
        
        # find total sentence number
        total_sentence_num = len(self.document[item].sentences)

        return (total_sentence_num - i) / total_sentence_num

    def _inter_sentence_similarity(self, item, i, j):
        '''
        Return inter sentence similarity score
        score = |Intersection (words of sentence, words of other sentence)| 
                / |Union (words of sentence, words of other sentence)|

        Parameter: 
        i - sentence index in paragraph
        j - other sentence index in paragraph 
        item - the paragraph in the paper
        '''

        sentence1 = self.document[item].sentences[i]
        sentence2 = self.document[item].sentences[j]

        intersection_total = 0
        for word1 in sentence1.words:
            if sentence2.words.count(word1) > 0: 
                intersection_total += sentence2.words.count(word1)
            else:
                for word2 in sentence2.words:
                    if word1 in self.word_to_id and word2 in self.word_to_id:
                        word1_id = self.word_to_id[word1]
                        word2_id = self.word_to_id[word2]
                        intersection_total += self.word_sim[word1_id][word2_id]

        return intersection_total / (len(sentence1.words) + len(sentence2.words))

    def _proper_nouns(self, item, i):
        '''
        Return proper noun score
        score = proper_noun_num / sentence_length
        
        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''
        sentence = self.document[item].sentences[i]

        sentence_length = len(sentence.words)

        tags = pos_tag(sentence.words)

        proper_noun_num = 0
        for tag in tags:
            if tag[1] == 'NNP':
                proper_noun_num += 1

        return proper_noun_num / sentence_length

    def _thematic_word(self, item, i):
        '''
        Return thematic word score
        score = thematic_words_in_sentence / total_thematic_words
        
        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''
        sentence = self.document[item].sentences[i]

        thematic_words_in_sentence = 0

        for keyword in self.thematic_words[item]:

            if sentence.words.count(keyword) > 0: 
                thematic_words_in_sentence += sentence.words.count(keyword)
            else:
                for word in sentence.words:
                    if word in self.word_to_id and keyword in self.word_to_id:
                        keyword_id = self.word_to_id[keyword]
                        word_id = self.word_to_id[word]
                        thematic_words_in_sentence += self.word_sim[keyword_id][word_id]
        return thematic_words_in_sentence / (self.total_thematic_words[item] + 1)


    def _numerical_data(self, item, i):
        '''
        Return numerical data score
        score = number of numerical data / length of sentence\

        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''
        sentence = self.document[item].sentences[i]
        num_numerical_data = 0

        for word in sentence.words:
            if self._is_number(word):
                num_numerical_data += 1

        return num_numerical_data / len(sentence.words)
    
    def _keywords(self, item, i):
        '''
        Return keyword score
        score = sum of all keywords {number of keyword * weight of keyword} / num_words in sentence

        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''
        sentence = self.document[item].sentences[i]

        total = np.zeros(len(self.keywords))

        for j in range(len(self.keywords)):
            keyword = self.keywords[j]
            keyword_weight = self.keywords_weights[j]
            if sentence.words.count(keyword) > 0: 
                total[j] += sentence.words.count(keyword)
            else:
                for word in sentence.words:
                    if word in self.word_to_id and keyword in self.word_to_id:
                        keyword_id = self.word_to_id[keyword]
                        word_id = self.word_to_id[word]
                        total[j] += self.word_sim[keyword_id][word_id]

            total += sentence.words.count(keyword) * keyword_weight

        return np.dot(total, self.keywords_weights.transpose()) / len(sentence.words)

    def _calculate(self, item, i):
        '''
        Return total score of a sentence 

        Parameters:
        i - sentence index in paragraph
        item - the paragraph in the paper
        '''
        sentence = self.document[item].sentences[i]

        scores = np.zeros(8)

        scores[0] = self._thematic_word(item, i)
        scores[1] = self._sentence_length(item, i)
        scores[2] = self._sentence_position(item, i)

        # inter sentence similarity 
        total = 0
        for j in range(len(self.document[item].sentences)):
            if i != j:
                total += self._inter_sentence_similarity(item, i, j)
        scores[3] = total / (len(self.document[item].sentences) - 1) 

        scores[4] = self._proper_nouns(item, i)
        scores[5] = self._thematic_word(item, i)
        scores[6] = self._numerical_data(item, i)
        scores[7] = self._keywords(item, i)

        total_score = np.dot(scores, self.score_weights)
        return scores, total_score

    def retrieve_document_names(self):
        '''
        Return heading of documents
        '''
        return list(self.document.keys())

    def set_thematic_threshold(self, new_threshold):
        '''
        Set thematic_threshold
        '''
        self.thematic_threshold = new_threshold

    def _prep_query_words(self, query):
        '''
        Return word similarity for query words
        '''
        # get all unique word from document
        query_words = defaultdict()

        # add words from title and keywords
        for word in query: 
            if word not in query_words and word in self.word_vectors.vocab and word not in self.word_to_id:
                query_words[word] = self.word_vectors[word]

        # set id retrieval based on word to help calculate word similarity
        word_to_id = defaultdict()

        words = list(query_words.keys())
        for word_id in range(len(words)):
            word = words[word_id]
            word_to_id[word] = word_id

        word_vecs = []
        for word in query_words:
            word_vecs.append(query_words[word])

        if len(word_vecs) == 0:
            return (np.array([]), word_to_id)

        cos_sim_nominator = np.dot(word_vecs, np.transpose(self.relevant_word_vector))

        relevant_word_vector_mag = np.linalg.norm(self.relevant_word_vector, axis=1)
        word_vec_mags = np.linalg.norm(word_vecs, axis=1)

        sim = np.transpose(cos_sim_nominator / relevant_word_vector_mag) / word_vec_mags 

        sim = np.transpose(sim)

        return (sim, word_to_id)

    def _query_word(self, item, i, query):
        '''
        Return sentence score based on query word

        Parameter: 
        i - sentence index in paragraph
        item - the paragraph in the paper
        query - list of query words
        '''
        (query_word_sim, query_words_to_id) = self._prep_query_words(query)

        sentence = self.document[item].sentences[i]
        intersection_total = 0
        for keyword in query:
            if sentence.words.count(keyword) > 0: 
                intersection_total += sentence.words.count(keyword)
            else:
                for word in sentence.words:
                    if word in self.word_to_id and keyword in self.word_to_id:
                        keyword_id = self.word_to_id[keyword]
                        word_id = self.word_to_id[word]
                        intersection_total += self.word_sim[keyword_id][word_id]
                    elif word in self.word_to_id and keyword in query_words_to_id:
                        keyword_id = query_words_to_id[keyword]
                        word_id = self.word_to_id[word]
                        intersection_total += query_word_sim[keyword_id][word_id]

            
        return intersection_total / (len(query)+len(sentence.words))


    def calculate(self):
        '''
        Return total score of a section in a document
        '''

        self.total_scores = []
        self.score_matrices = []
        for item in self.document:
            sentence_scores = []
            score_matrix = []
            for i in range(len(self.document[item].sentences)):
                res = self._calculate(item, i)
                sentence_scores.append(res[1])
                score_matrix.append(res[0])
            self.total_scores.append(sentence_scores)
            self.score_matrices.append(score_matrix)
        return (self.score_matrices, self.total_scores)

    def query(self, query_string):
        '''
        Return total score of sentences based on user query and sentence features
        '''
        query = query_string.lower().split()
        total_scores = []
        for item in self.document:
            sentence_scores = []
            for i in range(len(self.document[item].sentences)):
                res = self._query_word(item, i, query)
                sentence_scores.append(res)
            total_scores.append(sentence_scores)

        return total_scores




