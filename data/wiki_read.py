# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:08:01 2020

@author: Matteo
"""

import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import stem
import re
#%% Train data
with open(r'C:\Users\Matteo\Documents\Git\aLDA\data\wikitext-2-raw\wiki.train.raw', encoding="utf8") as file:  
    dataRaw = file.read()

dataRaw = stem(dataRaw)
data = dataRaw.replace('= = = = ','+ + + +')
data = data.replace('= = =','- - -')
data = data.replace('= =','* *')

data = data.split(" = ")
datagensim = []
regex = re.compile('[^a-zA-Z ]')
for d in data:

    #First parameter is the replacement, second parameter is your input string
    test = regex.sub('', d)
    #Out: 'abdE'
    if len(test)>100:
        datagensim += [[i for i in test.split(" ") if len(i)>3]]

dct = Dictionary(datagensim)
dct.filter_extremes(no_below=10, no_above=0.5 )
dct.compactify()
X = np.zeros((len(dct.keys()),len(datagensim)),int)
i = 0
bow = []
for d in datagensim:
    tmp = dct.doc2bow(d)
    bow += [tmp] 
    for key, value in tmp:
        X[key,i] = value
    i +=1
##%% Test data
#with open(r'C:\Users\Matteo\Documents\Git\aLDA\data\wikitext-2-raw\wiki.test.raw', encoding="utf8") as file:  
#    dataRaw = file.read()
#
#dataRaw = stem(dataRaw)
#data = dataRaw.replace('= = = = ','+ + + +')
#data = data.replace('= = =','- - -')
#data = data.replace('= =','* *')
#
#data = data.split(" = ")
#datagensim_test = []
#regex = re.compile('[^a-zA-Z ]')
#for d in data:
#
#    #First parameter is the replacement, second parameter is your input string
#    test = regex.sub('', d)
#    #Out: 'abdE'
#    if len(test)>100:
#        datagensim_test += [[i for i in test.split(" ") if len(i)>3]]
#
##dct_test = Dictionary(datagensim_test)
##dct_test.filter_extremes(no_below=10, no_above=0.5 )
##dct_test.compactify()
#X_test = np.zeros((len(dct.keys()),len(datagensim_test)),int)
#i = 0
#bow_test = []
#for d in datagensim_test:
#    tmp = dct.doc2bow(d)
#    bow_test += [tmp] 
#    for key, value in tmp:
#        X_test[key,i] = value
#    i +=1

