# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:08:01 2020

@author: Matteo
"""

import numpy as np
import string
from sklearn import preprocessing
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#%%
with open(r'C:\Users\Matteo\Documents\Git\aLDA\data\wikitext-2-raw\wiki.train.raw', encoding="utf8") as file:  
    data = file.read()
    
    
    
#%%
data = data.replace('= = = = ','+ + + +')
data = data.replace('= = =','- - -')
data = data.replace('= =','* *')

data = data.split(" = ")

#%%
stopset = stopwords.words('english') 
unwantedchar = string.punctuation + string.digits


vectorizer = CountVectorizer( encoding='utf-8', decode_error='strict', strip_accents='ascii', lowercase=True, analyzer='word', max_df=0.8, min_df=4)
X = vectorizer.fit_transform(data)

#%%




