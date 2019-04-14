# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:30:24 2019

@author: Jigyasa Yadav
"""

from nltk.stem.porter import *

#Create a new Porter stemmer.

stemmer = PorterStemmer()
#Test the stemmer on various pluralised words.

plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
            'died', 'agreed', 'owned', 'humbled', 'sized',
            'meeting', 'stating', 'siezing', 'itemization',
            'sensational', 'traditional', 'reference', 'colonizer',
          'plotted']
singles = [stemmer.stem(plural) for plural in plurals]
print(' '.join(singles))  # doctest: +NORMALIZE_WHITESPACE