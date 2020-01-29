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

d_max = 800

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
max_N = max([len(x) for x in corpusId])
WTrain = np.zeros((max_N,d_max)) -1
WTest = np.zeros((max_N,d_max)) -1
AMask = np.zeros((nb_a,d_max))
W_train = []
W_test = []
for d in range(d_max):
      nbTrain = int(3*len(corpusId[d])/4)
      l_ = np.arange(len(corpusId[d]))
      np.random.shuffle(l_)
      WTrain[:nbTrain,d] = corpusId[d][l_[:nbTrain]]
      WTest[:(len(corpusId[d])-nbTrain),d] = corpusId[d][l_[nbTrain:]]
      AMask[authorDocId[d],d] = 1
      W_train.append([(w,np.sum(WTrain[:,d] == w)) for w in range(n_dic)])
      W_test.append([(w,np.sum(WTest[:,d] == w)) for w in range(n_dic)])
Adic = {}
for a in  range(nb_a):
      Adic[str(a)] = list(np.where(AMask[a,:]>0)[0])
#%%
alpha = 1.8
beta = 1.8
gamma = -1
K = 20
aLDA_train = aLDA_estimator(K, WTrain, AMask, alpha, beta, gamma)
aLDA_train.gd_ll(0.008, 100, 0,0.0,0,0)
plt.plot(aLDA_train.llgd/aLDA_train.llgd[0,:])

#%%

from gensim.models import AuthorTopicModel

model = AuthorTopicModel(W_train, author2doc=Adic,  num_topics=K)
#%%
phiGen = model.get_topics().transpose()
thetaGen = 0*aLDA_train.thetaStar
for a in  range(nb_a):
      thetaGen[:,a] = [b for (c,b) in model.get_author_topics(str(a),0)]
#%%
      
aLDA_test = aLDA_estimator(K, WTest, AMask, alpha, beta, gamma) 
#%%
print('ll + Pa + Pb / Learning set')

#print(loglikaLDA(thetaStar, phiStar, AStar, D, alpha, beta))   
print(loglikaLDA(aLDA_train.thetaStar, aLDA_train.phiStar, aLDA_train.AStar, aLDA_test.D, alpha, beta,0))        
print(loglikaLDA(thetaGen, phiGen, AMask/np.sum(AMask,0), aLDA_test.D, alpha, beta,0))           

  