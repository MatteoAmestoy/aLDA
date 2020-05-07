# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:16:05 2020

@author: Matteo
"""

import numpy as np
from sklearn.preprocessing import normalize

#%%
      
class aLDA_generator():
      def __init__(self, n_dic, n_w, A_mask, K, alpha, beta, gamma):
            '''
            Input:
            - n_d = [int] nb of documents
            - n_dic = [int] nb of words in the dictionnary
            - n_w = [int] nb of words in a document
            - A_mask = [nb_a,n_d int] nb authors * nb doc matrix of author participation to each paper
                  (1 if author participated to paper)
            - K = [int] nb of topics
            - alpha[n_a float] distribution of theta (Dirichlet parameter)
            - beta [n_dic float] distribution of phi (Dirichlet parameter)
            - gamma [flaot] prior on A, -1 if uniform
            '''
            self.A_mask = A_mask
            self.K = K 
            self.n_dic = n_dic
            self.n_w = n_w
            self.n_a,self.n_d =  A_mask.shape
            if np.size(alpha) == 1:
                  self.alpha = alpha*np.ones(self.K) # [float] prior theta
            elif np.size(alpha) == self.K:
                  self.alpha = alpha
            else:
                  print('alpha error size (should be 1 or K)')
            if np.size(beta) == 1:
                  self.beta = beta*np.ones(self.n_dic) # [float] prior phi
            elif np.size(beta) == self.n_dic:
                  self.beta = beta # [float] prior phi
            else:
                  print('alpha error size (should be 1 or N)')           
            self.gamma = gamma

      def itialise(self):
            '''
            Initialises the values of A, theta, phi.

            '''
            # generate A
            if self.gamma == -1: # if we want same contribution from each author
                  self.A = normalize(self.A_mask,'l1',0)
            else:
                  self.A = np.zeros((self.n_a,self.n_d)) 
                  for d in range(self.n_d): # Standard Dirichlet
                        self.A[self.A_mask[:,d]==1,d] = np.random.dirichlet(np.ones(int(np.sum(self.A_mask[:,d])))*self.gamma)
                        
            # generate theta phi
            self.theta = np.random.dirichlet(self.alpha,(self.n_a)).transpose()
            self.phi = np.random.dirichlet(self.beta,(K)).transpose()  
            self.thetaDoc =self.theta.dot(self.A) # doc topic distribution
      def generate(self):
            '''
            Generate the corpus
            Output:
                  - C_ Corpus as a list of documents= list of pairs (word, count) 
                  - Z [n_w,n_d int] # Mat of topics
                  - C [n_w,n_d int] # Mat corpus
                  - D [n_dic,n_d int] # Mat word count
            '''            
            #----------- Doc generation
            C_ = [] # Corpus as a list of documents= list of pairs (word, count) 
            Z = np.zeros((self.n_w,self.n_d)).astype(int) # Mat of topics
            C = np.zeros((self.n_w,self.n_d)).astype(int) # Mat corpus
            D = np.zeros((self.n_dic,self.n_d)).astype(int) # Mat word count
            for d in range(self.n_d):
                  # generate number words in doc
                  n_ = self.n_w
                  # generate words
                  Z[:,d] = np.random.multinomial(1,self.thetaDoc[:,d],n_).argmax(1).astype(int)
                  for w in range(n_):
                        # Store topic|
                        C[w,d] = int(np.random.multinomial(1,self.phi[:,int(Z[w,d])]).argmax())
                        D[C[w,d],d] += 1
                  C_.append([(k,D[k,d]) for k in range(self.n_dic)])
            return(Z,C,D,C_)

#%% Test
n_dic = 2000
n_doc = 3000
n_w = 100
A_Mask = np.eye(n_doc)
K = 50
gamma = -1
alpha = 1
beta = 1
synt_LDA = aLDA_generator(n_dic, n_w, A_Mask, K, alpha, beta, gamma)
synt_LDA.itialise()
Z,C,D,C_ = synt_LDA.generate()


