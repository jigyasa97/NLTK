# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:40:51 2019

@author: Jigyasa Yadav
"""
from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "Hello mR. Sharma, how are you doing today? The weather is great and pythin is best.The sky is the limit.And now i should stop typing."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))

'''
['Hello mR. Sharma, how are you doing today?', 'The weather is great and pythin is best.The sky is the limit.And now i should stop typing.']
['Hello', 'mR.', 'Sharma', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'The', 'weather', 'is', 'great', 'and', 'pythin', 'is', 'best.The', 'sky', 'is', 'the', 'limit.And', 'now', 'i', 'should', 'stop', 'typing', '.']
'''