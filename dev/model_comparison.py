# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:22:35 2020

@author: Matteo
"""

import numpy as np

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
        for m in self.models:
            m.train()
        return()            

