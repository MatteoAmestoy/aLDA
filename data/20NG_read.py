# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:14:59 2020

@author: Matteo
"""

from sklearn.datasets import fetch_20newsgroups
import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import stem
from gensim.models import AuthorTopicModel,LdaModel,LdaMulticore
import re
#%% Importing directly from raw text
newsgroups_train = fetch_20newsgroups(subset='train', remove = 'headers')
dataRaw = newsgroups_train.data
data = [stem(i) for i in dataRaw]

datagensim = []
regex = re.compile('[^a-zA-Z ]')
for d in data[:200]:
    

    #First parameter is the replacement, second parameter is your input string
    test = regex.sub('', d)
    #Out: 'abdE'
    if len(test)>100:
        datagensim += [[i.lower() for i in test.split(" ") if len(i)>2]]
#gensim.utils.lemmatize(
dct = Dictionary(datagensim)
dct.filter_extremes(keep_n=50000, no_above=0.8 )
dct.compactify()
X = np.zeros((len(dct.keys()),len(datagensim)),int)
i = 0
bow = []
datagensimClean = []
for d in datagensim:
    
    idx = dct.doc2idx(d)
    dC = [d[i] for i in range(len(d)) if idx[i]>-1]
    tmp = dct.doc2bow(dC)
    datagensimClean += [dC]
    bow += [tmp] 
    for key, value in tmp:
        X[key,i] = value
    i +=1
    
datagensim =  datagensimClean 



#%%


common_corpus = [dct.doc2bow(text) for text in datagensim]

# Train the model on the corpus.



lda = LdaModel(common_corpus, num_topics=50, passes = 100)


#%%
aaa = CoherenceModel( lda, texts = datagensim, dictionary=dct,coherence='c_npmi',window_size=40,topn = 5)
aaa.get_coherence()


#%% Building from https://github.com/akashgit/autoencoding_vi_for_topic_models
import pickle as pk

dataAkash = np.load(r'C:\Users\Matteo\Desktop\autoencoding_vi_for_topic_models-master\autoencoding_vi_for_topic_models-master\data\20news_clean\train.txt.npy', encoding="bytes")
dataAkashTest = np.load(r'C:\Users\Matteo\Desktop\autoencoding_vi_for_topic_models-master\autoencoding_vi_for_topic_models-master\data\20news_clean\test.txt.npy', encoding="bytes")
dct = pk.load( open( r'C:\Users\Matteo\Desktop\autoencoding_vi_for_topic_models-master\autoencoding_vi_for_topic_models-master\data\20news_clean\vocab.pkl', "rb" ))
inv_dct = {v: k for k, v in dct.items()}

# build text document
dataAkashText = []

i = 0
for d in dataAkash:
    tmp = []
    for w in d:
        tmp += [inv_dct[w]]
    i +=1
    dataAkashText += [tmp]
dataAkashTextTest = []
i = 0
for d in dataAkashTest:
    tmp = []
    for w in d:
        tmp += [inv_dct[w]]
    i +=1
    dataAkashTextTest += [tmp]
    
    
    
dct = Dictionary(dataAkashText)

dataAkashIdx = []
# build count matrix
X = np.zeros((len(dct.keys()),len(dataAkashText)),int)
i = 0
for d in dataAkashText:
    
    idx = dct.doc2idx(d)
    dataAkashIdx += [idx]
    tmp = dct.doc2bow(d)
    for key, value in tmp:
        X[key,i] = value
    i +=1


dataAkashTestIdx = []
# build count matrix
XTest = np.zeros((len(dct.keys()),len(dataAkashTextTest)),int)
i = 0
for d in dataAkashTextTest:
    
    idx = dct.doc2idx(d)
    dataAkashTestIdx += [idx]
    tmp = dct.doc2bow(d)
    for key, value in tmp:
        XTest[key,i] = value
    i +=1

  
#%%

common_corpus = [dct.doc2bow(text) for text in dataAkashText]

# Train the model on the corpus.



lda = LdaModel(common_corpus, num_topics=50, passes = 100)





#%%
aaa = CoherenceModel( lda, texts = dataAkashTextTest, dictionary=dct,coherence='c_npmi',window_size=100,topn = 5)
aaa.get_coherence()



