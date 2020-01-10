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
stopset = stopwords.words('english') 
unwantedchar = string.punctuation + string.digits

corpus = []
all_docs = []
vocab = set()

stemmer = PorterStemmer()
i = 0
for doc in data_df_journals['abstract']:
      i += 1
      if i< 6000:
            doc_ = doc.translate(str.maketrans(unwantedchar,' '*len(unwantedchar)))           
            doc_ = [x for x in doc_.split() if len(x)>3] 
            doc_ = [x.lower() for x in doc_ if x not in stopset] 
            vocab.update(doc_)
            corpus.append(doc_)
vocab = list(vocab)
#%%
vocab2id = preprocessing.LabelEncoder()
vocab2id.fit(vocab)

vocabId = vocab2id.transform(vocab)
corpusId = [ vocab2id.transform(x) for x in corpus]

#%%

  