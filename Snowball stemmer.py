# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:20:37 2019

@author: Jigyasa Yadav
"""

from nltk.stem.snowball import SnowballStemmer
print(" ".join(SnowballStemmer.languages))


#stem a word
stemmer = SnowballStemmer("english")
print(stemmer.stem("running"))


#Decide not to stem stopwords. 
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
print(stemmer.stem("having"))

print(stemmer2.stem("having"))

#The 'english' stemmer is better than the original 'porter' stemmer.
print(SnowballStemmer("english").stem("generously"))

print(SnowballStemmer("porter").stem("generously"))
