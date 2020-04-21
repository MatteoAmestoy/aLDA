# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:22:35 2020

==========================================================================
Topic Modeling Comparison
==========================================================================

@author: Matteo Amestoy
Framework to compare multiple model in a unified environment
"""

import numpy as np
import time
from scipy.io import loadmat
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import scipy as sc
#%%

class model_comparison():
    def __init__(self, word_count, author_part, dct, models = [],testText =[]):
        '''
        Input:
        - word_count = [n_w,n_d int] matrix of the count of words for each document
        - author_part = [n_a,n_d bool] 1 if author i participated to doc j
        - dct = dic{int} maping of integers to words
        - testText = list of list of words to test the coherence
        '''
        self.D = word_count
        self.A = author_part 
        self.dct = dct
        self.n_w = self.D.shape[0]
        self.n_a,self.n_d =  self.A.shape
        self.models = models
        self.testText = testText

    def train_models(self):
        '''
        Train the models
        '''
        for m in self.models:
            t0 = time.time()
            print('--- Started training for '+m.name)
            m.train()
            print('finished after '+ str(time.time()-t0)+'seconds')
            print('----------------------------------------')
        return() 
           

    def compute_scores(self,l):
        for m in self.models:
            t0 = time.time()
            print('--- Scores for'+m.name)
            print('- Perplexity:')
            print(self.perplexity(m))
            print('- Topic Uniqueness (l='+str(l)+'):')
            print(self.topic_uniqueness(m,l))
            print('- Coherence NPMI (l='+str(l)+'):')
            print(self.coherence(m,l))
            print('finished after '+ str(time.time()-t0)+'seconds')
            print('----------------------------------------')
        return()
    
    def perplexity(self,model):
        '''
        Perplexity(log-likelihood)
        Input:
        - model = model class (fields used D_reb, estimated word probability matrix) 
        '''
        ll = np.sum(np.sum(np.log(model.D_reb)*self.D))
        return(ll)
    
    def coherence(self, model,L):
        test = np.argsort(model.phi,0)[-L:,:]
        topic = []
        for k in range(K):
            tmp = []
            for l in range(L):
                tmp += [self.dct[test[l,k]]]
            topic += [tmp]
        aaa = CoherenceModel( topics=topic, texts=self.testText, dictionary=dct, window_size=40,coherence='c_npmi', topn = 5)    

        return(aaa.get_coherence())
    
    def topic_uniqueness(self, model, L):
        '''
        Topic uniqueness cost function, that penalises topics that have top words not specific
        Input:
        - model = model class (fields used K number of topics, phi topic to word matrix)
        - L = number of top words that define a topic
        '''
        top_words  = np.zeros((self.n_w,model.K))
        top_words[np.argsort(model.phi,0)[-L:,:],:] = 1
        count = np.sum(top_words,1)
        count[count == 0] = 1 # if count = 0 then all the words is never a top word and 0/1=0 
        return(np.sum(top_words.T/count)/model.K/L)
        
    
#%% Nips Data

UM =  False
if UM:
    p = r'C:\Users\matteo.amestoy\Documents\DataSets\Nips_1-17\nips_1-17.mat'
else:
    p = r'C:\Users\Matteo\Documents\Git\aLDA\data\nips_1-17.mat'
        
x = loadmat(p)

M = x['counts']
A = x['docs_authors']
Words = x['words']
At = np.asarray(A.todense().T)
M_full = np.asarray(M.todense())

#M_full = M_full[:3000,:]
empty_idx = np.where(np.sum(M_full,0)==0)
M_full = np.delete(M_full,empty_idx,1)
At = np.delete(At,empty_idx,1)
n_dic,n_doc = M_full.shape
n_a = At.shape[0]
K = 50
#%% Wiki Data


M_full = X.T
At = np.eye(M_full.shape[1])
n_dic,n_doc = M_full.shape
n_a = At.shape[0]
K = 50

#np.sum(np.sum(np.log(phi.dot(theta).dot(A))*self.D))
#%% Train data

params ={}
params['train_param'] = {}


aTMm = aTM(K, M_full, At, params, 'aTM_baseline')
aTMm.train()


params['alpha'] = 1
params['beta'] = 1
params['gamma'] = 1
params['init_mat'] = {}
params['init_mat']['A'] = normalize(At,'l1',0)
params['init_mat']['theta'] = aTMm.theta
params['init_mat']['phi'] = aTMm.phi
params['train_param']['step']=0.0009
params['train_param']['n_itMax']= 70
params['train_param']['b_mom']=0.01
params['train_param']['X_priorStep']=0
params['train_param']['Y_priorStep']=0
params['train_param']['Z_step']=0

aLDATMm = aLDA_gd(K, M_full, At, params, 'aTM_gd_baseline')
aLDATMm.train()

#%%


params ={}
params['train_param'] = {}
LDAm = LDA(K, M_full, At, params, 'LDA_baseline')
LDAm.train()


params['alpha'] = 1
params['beta'] = 1
params['gamma'] = 1
params['init_mat'] = {}
params['init_mat']['A'] = np.eye(n_doc)
params['init_mat']['theta'] = LDAm.theta
params['init_mat']['phi'] = LDAm.phi
params['train_param']['step']=0.00008
params['train_param']['n_itMax']= 60
params['train_param']['b_mom']=0.01
params['train_param']['X_priorStep']=0
params['train_param']['Y_priorStep']=0
params['train_param']['Z_step']=0

aLDAm = aLDA_gd(K, M_full, np.eye(n_doc), params, 'aLDA_gd_baseline')
aLDAm.train()


#%%
Words = {}

#m = model_comparison(M_full, At, dct, models = [LDAm,aLDAm,aTMm,aLDATMm], testText = datagensim)   
#m.compute_scores(10)

#m = model_comparison(M_full, At, {}, models = [LDAm,aLDAm,aTMm,aLDATMm], testText = {})


   
#m.compute_scores(10)    
#m = model_comparison(M_full, At, dct, models = [LDAm,aLDAm], testText = datagensim)   
#m.compute_scores(10)

m = model_comparison(M_full, At, dct, models = [LDAm,aLDAm,aTMm,aLDATMm], testText = datagensim)   
m.compute_scores(10)

#m = model_comparison(M_full, At, dct, models = [LDAm,aLDAm,aTMm,aLDATMm], testText = datagensim_test)   
#m.compute_scores(10)




#%% Draft check learning

plt.subplot(1,2,1)
plt.plot(aLDAm.llgd[:,0])
plt.subplot(1,2,2)
plt.plot(aLDATMm.llgd[:,0])

#%% visualize topics
import pyLDAvis.gensim
import pyLDAvis# Visualize the topics


LDAvis_prepared = pyLDAvis.gensim.prepare(LDAm.LDA,bow,dct)
pyLDAvis.show(LDAvis_prepared)


#%%

dct.add_documents(common_corpus)

