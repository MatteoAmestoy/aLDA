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
            if len(aut)>0:
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

autId = aut2id.transform(authors)
authorDocId = [ aut2id.transform(x) for x in authorDoc]

#%%
n_dic = len(vocabId)
nb_a = len(autId)
max_len = max([len(x) for x in corpusId])
W = np.zeros((max_len,d_max)) -1
AMask = np.zeros((nb_a,d_max))
W_ = []
for d in range(d_max):
      W[:len(corpusId[d]),d] = corpusId[d]
      AMask[authorDocId[d],d] = 1
      W_.append([(w,np.sum(W[:,d] == w)) for w in range(n_dic)])
Adic = {}
for a in  range(nb_a):
      Adic[str(a)] = list(np.where(AMask[a,:]>0)[0])
#%%
alpha = 1.8
beta = 1.8
gamma = 500
K = 50
aaa = aLDA_estimator(K, W, AMask, alpha, beta, gamma)
aaa.gd_ll(0.008, 100, 0,0.0,0,0)
plt.plot(aaa.llgd/aaa.llgd[0,:])

#%%

from gensim.models import AuthorTopicModel

model = AuthorTopicModel(W_, author2doc=Adic,  num_topics=K)
#%%
phiGen = model.get_topics().transpose()
thetaGen = 0*aaa.thetaStar
for a in  range(nb_a):
      thetaGen[:,a] = [b for (c,b) in model.get_author_topics(str(a),0)]
#%%
#%%
print('ll + Pa + Pb / Learning set')

#print(loglikaLDA(thetaStar, phiStar, AStar, D, alpha, beta))   
print(loglikaLDA(aaa.thetaStar, aaa.phiStar, aaa.AStar, aaa.D, alpha, beta))        
print(loglikaLDA(thetaGen, phiGen, AMask/np.sum(AMask,0), aaa.D, alpha, beta))           

  