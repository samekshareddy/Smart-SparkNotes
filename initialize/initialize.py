# imports needed and logging
import gzip
import gensim 
import logging
import pdb
import time

from collections import defaultdict


from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

print('initializing ...')
t0 = time.time()
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, )

word_vectors = model.wv

fname = get_tmpfile("vectors.kv")
word_vectors.save(fname)


t1 = time.time()

print('done', t1-t0)