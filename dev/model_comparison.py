# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:22:35 2020

@author: Matteo
"""

import numpy as np
import time
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
        self.n_w = self.D[0]
        self.n_a,self.n_d =  self.A
        self.models = models

    def train_models(self):
        '''
        Train the models
        '''
        for m in self.models:
            t0 = time.time()
            print('--- Started training for '+m.name)
            m.train(self.D, self.A, m.param)
            print('finished after '+ str(time.time()-t0)+'seconds')
            print('----------------------------------------')
        return() 
           

    def compute_scores(self,scoreList):
        
        return()
    
    def perplexity(self,model):
        
        return()
    
    def coherence(self, model):
        
        return()
    
    def topic_uniqueness(self, model, L):
        
        top_words  = np.zeros((self.n_w,model.K))
        for k in range(model.k):
            top_words[np.argsort(model.phi)[-L:],k] = 1
        count = np.sum(top_words,0)
        return(np.sum(top_words/count)/model.K/L)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    