# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:10:48 2019

@author: Jigyasa Yadav
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "this is an example to test stop words filteration."
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

filtered_sentences = []

for w in words:
    if w not in stop_words:
        filtered_sentences.append(w)
        
print(filtered_sentences)
            
            
#another way of writing
filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)