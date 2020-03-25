# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:15:10 2020

==========================================================================
Topic Extraction Models
==========================================================================

@author: Matteo Amestoy
Adapt all the topic extraction models to fit the model_comparison script
"""

import numpy as np
import sys
from sklearn.preprocessing import normalize
from gensim.models import AuthorTopicModel,LdaModel

#%%

class aLDA_gd():
    def __init__(self, K, data, AMask, params, name):
        '''
        Input:
        - K = [int] nb of topics
        - data = [n_dic,n_d int] size of dictionary * nb doc matrix of word count
        - AMask = [n_a,n_d 0-1] nb authors * nb doc matrix of author participation to each paper
              (1 if author participated to paper)
        - params dic with fields
            - alpha[n_a float] priors on theta
            - beta [n_dic float] priors on  phi
            - gamma [flaot] prior on A
            - init_mat dic with fields TBW
        - name = char name of the model
        '''        
        self.K = K # [int] nb of topics
        self.AMask = AMask # [n_a,n_d float] matrix of author participation to each paper (1 if author participated to paper)
        self.n_a,self.n_d = self.AMask.shape # [int] nb authors
        self.D = data
        self.n_dic,self.n_d = self.D.shape    
        self.name = name

        if np.size(params.alpha) == 1:
              self.alpha = params.alpha*np.ones(self.K) # [float] prior theta
        elif np.size(params.alpha) == self.K:
              self.alpha = params.alpha
        else:
              print('alpha error size (should be 1 or K)')
        if np.size(params.beta) == 1:
              self.beta = params.beta*np.ones(self.n_dic) # [float] prior phi
        elif np.size(params.beta) == self.n_dic:
              self.beta = params.beta # [float] prior phi
        else:
              print('alpha error size (should be 1 or N)')           
        self.gamma = params.gamma
        self.init_mat = params.init_mat
        self.train_param = params.train_param
        
        
    def loglik(self, theta, phi, A):
        '''
        Computes the log likelihood and priors of a given a set of parameters.
        Input:
              - theta [K,n_a float] each column is the distribution of topics for an author
              - phi [n_dic,K float] each column is the distribution of words for a thopic
              - A [n_a,n_d float] each column is the distribution of authors for a document
        '''
        M = np.sum(np.sum(np.log(phi.dot(theta).dot(A))*self.D)) # Likelihood
        pTheta = np.sum((self.alpha - 1).dot(np.log(theta))) # Dirichlet prior 
        pPhi = np.sum((self.beta - 1).dot(np.log(phi))) # Dirichlet prior 
        pA = 1#(self.gamma - 1)*np.sum(np.log(A**self.AMask)) # Dirichlet prior constant gamma
        return(M, pTheta, pPhi, pA)
    
    def init_gd(self):
        '''
        Initialisation of the gradient descent
        '''
        dic_keys = self.init_mat.keys()
        if 'theta' in dic_keys:
              X = np.log(self.init_mat['theta']+sys.float_info.epsilon)
        else:
              X = np.random.normal(0,1,(self.K,self.n_a))  
        if 'phi' in dic_keys:
              Y = np.log(self.init_mat['phi']+sys.float_info.epsilon)
        else:
              Y = np.random.normal(0,1,(self.n_dic,self.K))
        if 'A' in dic_keys:
              Z = np.log(self.init_mat['A']+sys.float_info.epsilon)
        else:
              Z = np.random.normal(0,1,(self.AMask.shape))
        return(X,Y,Z)

    def gd_ll(self, step, n_itMax, b_mom , X_priorStep, Y_priorStep, Z_step):
        '''
        Gradient Descent optimisation of the aLDA loglikelihood
        TO BE DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        stability of sigmoid function
        tolerance value to stop automatically
        '''
        # Utility variables -----------------------------------------------
        self.llgd = np.zeros((n_itMax,4))
        
        X,Y,Z = self.init_gd()
        
        theta = normalize(np.exp(X),'l1',0)
        phi = normalize(np.exp(Y),'l1',0)
        A = normalize(self.AMask*np.exp(Z),'l1',0)
        
        VX = np.zeros((self.K,self.n_a))
        VY = np.zeros((self.n_dic,self.K))
        VZ = np.zeros(self.AMask.shape)
        
        self.llgd[0,:] = self.loglik(theta,phi,A)
    
        # gradient descent to maximize the posterior likelihood ----------- 
        for it in range(n_itMax-1): 
              # Ratio matrix (\tilde{D} in paper)
              Dg = self.D/(phi.dot(theta.dot(A)))
              
              # Compute gradient and update
              dX = theta*(phi.T.dot(Dg.dot(A.T))-np.diag(A.dot(Dg.T.dot(phi.dot(theta)))))+X_priorStep*(self.alpha[:, None]-1-theta*np.sum(self.alpha-1))
              VX = b_mom*VX + (1-b_mom)*dX
              X = X + step*VX
              theta = normalize(np.exp(X),'l1',0)
              
              dY = phi*(Dg.dot(A.T.dot(theta.T))-np.diag(phi.T.dot(Dg.dot(A.T.dot(theta.T)))))+Y_priorStep*(self.beta[:, None]-1-phi*np.sum(self.beta-1))
              VY = b_mom*VY + (1-b_mom)*dY
              Y = Y + step*VY
              phi = normalize(np.exp(Y),'l1',0)
              
              
              dZ = A*(theta.T.dot(phi.T.dot(Dg)-np.diag(Dg.T.dot(phi.dot(theta.dot(A))))))
              VZ = b_mom*VZ + (1-b_mom)*dZ+1*((self.gamma-1)*(1-A*np.sum(self.AMask,0)))
              Z = Z + Z_step*VZ    
              A = normalize(self.AMask*np.exp(Z),'l1',0) 
              
              # Store ll
              self.llgd[it+1,:] = self.loglik(theta,phi,A)

                    
        # Store estimates
        self.theta = theta
        self.phi = phi
        self.A = A
        self.D_reb = phi.dot(theta).dot(A)
    def train(self, p):
        self.gd_ll(p.step, p.n_itMax, p.b_mom , p.X_priorStep, p.Y_priorStep, p.Z_step)    
        return()

#%% LDA



class LDA():
    def __init__(self, K, data, AMask, params, name):
        self.K = K # [int] nb of topics
        self.AMask = AMask # [n_a,n_d float] matrix of author participation to each paper (1 if author participated to paper)
        self.n_a,self.n_d = self.AMask.shape # [int] nb authors
        self.D = data
        self.n_dic,self.n_d = self.D.shape    
        self.name = name
        self.train_C_ = []
        for d in range(self.n_d):
              self.train_C_.append([(k,self.D[self.K,d]) for k in range(self.n_dic)])
        self.LDA = LdaModel(self.train_C_, num_topics=self.K)


    def train(self, p):       
        self.phi = self.LDA.get_topics().transpose()
        self.theta = np.zeros((self.K,self.n_d))
        for d in  range(self.n_d):
            tmp = self.LDA.get_document_topics(self.train_C_[d])
            ind = [c for (c,b) in tmp]
            self.theta[ind,d] = [b for (c,b) in tmp]
        self.D_reb = self.phi.dot(self.theta)          
        return()
    
#%% aTM

class aTM():
    def __init__(self, K, data, AMask, params, name):
        self.K = K # [int] nb of topics
        self.AMask = AMask # [n_a,n_d float] matrix of author participation to each paper (1 if author participated to paper)
        self.n_a,self.n_d = self.AMask.shape # [int] nb authors
        self.D = data
        self.n_dic,self.n_d = self.D.shape    
        self.name = name
        
        self.train_C_ = []
        for d in range(self.n_d):
              self.train_C_.append([(k,self.D[self.K,d]) for k in range(self.n_dic)])
        
        Adic = {}
        for a in  range(self.n_a):
            Adic[str(a)] = list(np.where(self.AMask[a,:]>0)[0])
            
        self.aTM = AuthorTopicModel(self.train_C_ , author2doc=Adic, num_topics=self.K)
        


    def train(self, p):     
        
        self.phi = aTM.get_topics().transpose()
        self.theta = np.zeros((self.K,self.n_a))
        for a in  range(self.n_a):
            self.theta[:,a] = [b for (c,b) in self.aTM.get_author_topics(str(a),0)]
        
        self.D_reb = self.phi.dot(self.theta).dot(normalize(self.AMask,'l1',0))    






