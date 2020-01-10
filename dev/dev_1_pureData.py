# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:05:54 2020

@author: admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
from sklearn import preprocessing
from nltk.stem.porter import *
from nltk.corpus import stopwords
#%% Load data
path = r'C:\Users\admin\Documents\Data_PURE\PURE_data_Jerry\PURE_data_Jerry'
data_df_journals = pd.read_pickle(path + r'\journalDB.pkl')
n_journals = len(data_df_journals['department'])    
data_df_authors = pd.read_pickle(path + r'\authorsDB.pkl')
n_authors = len(data_df_authors['unitList'])

#%%

d_max = 500

stopset = stopwords.words('english') 
unwantedchar = string.punctuation + string.digits

corpus = []
authorDoc = []

vocab = set()
authors = set()
stemmer = PorterStemmer()
i = 0
for index,row in data_df_journals.iterrows():
      if i<d_max+1:
            doc = row['abstract']
            aut = [a for a in row['authorsID'] if a !=-1]
            doc_ = doc.translate(str.maketrans(unwantedchar,' '*len(unwantedchar)))           
            doc_ = [stemmer.stem(x) for x in doc_.split() if len(x)>5] 
            doc_ = [x.lower() for x in doc_ if x not in stopset] 
            vocab.update(doc_)
            corpus.append(doc_)
            
            authors.update(aut)
            authorDoc.append(aut)
            i +=1
vocab = list(vocab)
authors = list(authors)

#%%
vocab2id = preprocessing.LabelEncoder()
vocab2id.fit(vocab)

vocabId = vocab2id.transform(vocab)
corpusId = [ vocab2id.transform(x) for x in corpus]

aut2id = preprocessing.LabelEncoder()
aut2id.fit(authors)

autId = vocab2id.transform(authors)
authorDocId = [ vocab2id.transform(x) for x in authorDoc]

#%%
max_len = max([len(x) for x in corpusId])
D = np.zeros((max_len,d_max)) -1
AMask = 
for d in range(d_max):
      D[:len(corpusId[d]),d] = corpusId[d]








  