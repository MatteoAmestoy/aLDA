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
#%%

class model_comparison():
    def __init__(self, word_count, author_part, word_map, models = []):
        '''
        Input:
        - word_count = [n_w,n_d int] matrix of the count of words for each document
        - author_part = [n_a,n_d bool] 1 if author i participated to doc j
        - word_map = dic{int} maping of integers to words
        '''
        self.D = word_count
        self.A = author_part 
        self.dic = word_map
        self.n_w = self.D.shape[0]
        self.n_a,self.n_d =  self.A.shape
        self.models = models

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
    
    def coherence(self, model):
        
        return()
    
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
        
    
#%%

UM =  False
if UM:
    p = r'C:\Users\matteo.amestoy\Documents\DataSets\Nips_1-17\nips_1-17.mat'
else:
    p = r'C:\Users\Matteo\Documents\Git\aLDA\data\nips_1-17.mat'
        
x = loadmat(p)
#%%

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


#%%

params ={}
params['train_param'] = {}
aTMm = aTM(K, M_full, At, params, 'aTM_baseline')
aTMm.train()

#%%
LDAm = LDA(K, M_full, At, params, 'LDA_baseline')
LDAm.train()
#%%
params['alpha'] = 1
params['beta'] = 1
params['gamma'] = 1
params['init_mat'] = {}
params['init_mat']['A'] = normalize(At,'l1',0)
params['init_mat']['theta'] = aTMm.theta
params['init_mat']['phi'] = aTMm.phi
params['train_param']['step']=0.001
params['train_param']['n_itMax']= 40
params['train_param']['b_mom']=0.001
params['train_param']['X_priorStep']=0
params['train_param']['Y_priorStep']=0
params['train_param']['Z_step']=0

aLDATMm = aLDA_gd(K, M_full, At, params, 'aTM_gd_baseline')
aLDATMm.train()

#%%
params['alpha'] = 1
params['beta'] = 1
params['gamma'] = 1
params['init_mat'] = {}
params['init_mat']['A'] = np.eye(n_doc)
params['init_mat']['theta'] = LDAm.theta
params['init_mat']['phi'] = LDAm.phi
params['train_param']['step']=0.001
params['train_param']['n_itMax']= 40
params['train_param']['b_mom']=0.001
params['train_param']['X_priorStep']=0
params['train_param']['Y_priorStep']=0

aLDAm = aLDA_gd(K, M_full, np.eye(n_doc), params, 'aLDA_gd_baseline')
aLDAm.train()
 #%%

m = model_comparison(M_full, At, Words, models = [LDAm,aLDAm,aTMm,aLDATMm])   
m.compute_scores(5)

#%%
    
