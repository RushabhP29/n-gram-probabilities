import math

import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk import *
import os

corpus = open('corpus.txt').read()
tokens = WhitespaceTokenizer().tokenize(corpus)
bigrams = ngrams(tokens, 2)

total_words = len(tokens)
freq = nltk.FreqDist(tokens)
print('Top 15 Frequent Words in Unigram Model: \n'+str(freq.most_common(15)))
print('\n')
unigram_p = {}
bigram_p = {}
unigram = {}
bigram = {}
uni_count = 0
for word in tokens:
    uni_count += 1
    if word in unigram:
        unigram[word] += 1
    else:
        unigram[word] = 1

for word in unigram:
        temp = [word]
        unigram_p[tuple(temp)] = math.log(float(unigram[word])/uni_count, 2)

freq = nltk.FreqDist(bigrams)
print('Top 15 Frequent Words in Bigram Model: \n'+str(freq.most_common(15)))
print('\n')
print('Unigram Probabilities: ')
print(unigram_p)

bigram_tuples = tuple(nltk.bigrams(tokens))
for item in bigram_tuples:
            if item in bigram:
                bigram[item] += 1
            else:
                bigram[item] = 1

for word in bigram:
    bigram_p[tuple(word)] = math.log(float(bigram[word]) / unigram[word[0]], 2)

print('\n')
print('Bigram Probabilities: ')
print(bigram_p)
