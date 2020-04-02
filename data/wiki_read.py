# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:08:01 2020

@author: Matteo
"""

import numpy as np
from gensim.corpora import Dictionary
import re
#%%
with open(r'C:\Users\Matteo\Documents\Git\aLDA\data\wikitext-2-raw\wiki.train.raw', encoding="utf8") as file:  
    dataRaw = file.read()
    
#%%
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
        datagensim += [[i for i in test.split(" ") if len(i)>0]]
#%%

dct = Dictionary(datagensim)
dct.filter_extremes(no_below=10, no_above=0.4 )
dct.compactify()
X = np.zeros((len(dct.keys()),len(datagensim)),int)
i = 0
for d in datagensim:
    tmp = dct.doc2bow(d)
    for key, value in tmp:
        X[key,i] = value
    i +=1

