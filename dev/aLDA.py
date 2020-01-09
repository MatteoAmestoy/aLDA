# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:18:51 2019

==========================================================================
Author Latent Dirichlet Allocation with gradient descent
==========================================================================

@author: Matteo Amestoy
Little modification of the LDA to take into account that authors participate to
multiple papers.
Optimal parameters found by gradient descent. 
Formulae can be found in ???
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time

#%%
def sigmoid(x):                                        
      return 1 / (1 + np.exp(-x))

# needs to be upgraded to non constant beta/alpha
class aLDA_estimator():
      def __init__(self, K, data, A, alpha, beta):
            '''
            Input:
            - K = [int] nb of topics
            - data = [n_w,n_d int] nb words * nb doc matrix of word index
            - A = [n_a,n_d float] nb authors * nb doc matrix of author contribution to each paper
                  must sum to 1
            - alpha,beta [float] priors on theta and phi
            '''
            self.K = K # [int] nb of topics
            self.A = A # [n_a,n_d float] matrix of author contribution to each paper
            self.n_a = self.A.shape[0] # [int] nb authors 
            self.M = data # [n_w,n_d int] matrix of word index
            self.n_dic = int(data.max())+1 # [int] nb words in dictionary
            if np.size(alpha) == 1:
                  self.alpha = alpha*np.ones(K) # [float] prior theta
            elif np.size(alpha) == K:
                  self.alpha = alpha
            else:
                  print('alpha error size (should be 1 or K)')
            if np.size(beta) == 1:
                  self.beta = beta*np.ones(n_dic) # [float] prior phi
            elif np.size(beta) == n_dic:
                  self.beta = beta # [float] prior phi
            else:
                  print('alpha error size (should be 1 or N)')           
            
            
            self.n_w,self.n_d = data.shape # nb words per doc, nb doc
            
            self.D = np.zeros((self.n_dic,self.n_d)) # [n_dic,n_d int] matrix of count for each word 
            for w in range(self.n_dic):
                  self.D[w,:] = np.sum(self.M==w,0) 
            
                        
      def loglik(self, theta, phi):
            '''
            Computes the log likelihood and priors of a given a set of parameters.
            Input:
                  - theta [K,n_a float] each column is the distribution of topics for an author
                  - phi [n_dic,K float] each column is the distribution of words for a thopic
            '''
            M = np.sum(np.sum(np.log(phi.dot(theta).dot(self.A))*self.D)) # Likelihood
            ptheta = np.sum((self.alpha-1).dot(np.log(theta))) # Dirichlet prior 
            pPhi = np.sum((self.beta-1).dot(np.log(phi))) # Dirichlet prior 
#            pPhi = np.sum(np.log(np.sum(phi**self.beta,0))) # Strange prior to be checked
#            ptheta = np.sum(np.log(np.sum(theta**self.alpha,0))) # Strange prior to be checked
            return(M,ptheta,pPhi)
            
      def gd_ll(self, step, n_itMax, tolerance, b_mom , X0, Y0):
            '''
            Gradient Descent optimisation of the aLDA loglikelihood
            TO BE DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            put default values to all parameters
            initialisation options
            stability of sigmoid function
            step :
            '''
            # Utility variables -----------------------------------------------
            self.llgd = np.zeros((n_itMax,3))
            
            # initialize ------------------------------------------------------
            X = np.random.normal(0,4,(self.K,self.n_a))
            Y = np.random.normal(0,4,(self.n_dic,self.K))
#            X = X0
#            Y = Y0
#            theta = sigmoid(X)
#            phi = sigmoid(Y)
            theta = np.exp(X)
            phi = np.exp(Y)
#            theta = np.random.dirichlet(np.ones(self.K),(self.n_a)).transpose()
#            phi = np.random.dirichlet(np.ones(self.n_dic),(self.K)).transpose()
            thetaB = normalize(theta,'l1',0)
            phiB = normalize(phi,'l1',0)  
            VX = np.zeros((self.K,self.n_a))
            VY = np.zeros((self.n_dic,self.K))
            self.llgd[0,:] = self.loglik(thetaB,phiB)

            # gradient descent to maximize the posterior likelihood ----------- 
            for it in range(n_itMax-1):
                  
                  Dg = self.D/(phiB.dot(thetaB.dot(self.A)))
#                  dX = thetaB*(1-theta)*(phiB.transpose().dot(Dg.dot(self.A.T))-np.diag(self.A.dot(Dg.T.dot(phiB.dot(thetaB)))))+self.alpha*(thetaB**alpha)/np.sum(thetaB**alpha,0)
#                  dY = phiB*(1-phi)*(Dg.dot(self.A.T.dot(thetaB.T))-np.diag(phiB.T.dot(Dg.dot(self.A.T.dot(thetaB.T)))))+self.alpha*(phiB**beta)/np.sum(phiB**beta,0)
                  dX = thetaB*(phiB.transpose().dot(Dg.dot(self.A.T))-np.diag(self.A.dot(Dg.T.dot(phiB.dot(thetaB)))))+30*(self.alpha[:, None]-1-thetaB*np.sum(self.alpha-1))
                  dY = phiB*(Dg.dot(self.A.T.dot(thetaB.T))-np.diag(phiB.T.dot(Dg.dot(self.A.T.dot(thetaB.T)))))+30*(self.beta[:, None]-1-phiB*np.sum(self.beta-1))
                  VX = b_mom*VX + (1-b_mom)*dX
                  VY = b_mom*VY + (1-b_mom)*dY
                  X = X + step*VX
                  Y = Y + step*VY
                  theta = np.exp(X)
                  phi = np.exp(Y)
                  thetaB = normalize(theta,'l1',0)
                  phiB = normalize(phi,'l1',0)    
                  self.llgd[it+1,:] = self.loglik(thetaB,phiB)
                  if np.mod(it+1,10) ==0:
                        print(it+1)
                        print(self.llgd[it+1]/self.llgd[it+1-10])
            self.thetaStar = thetaB
            self.phiStar = phiB

def loglikaLDA(theta, phi, A, D, alpha, beta):
      '''
      Computes the log likelihood and priors of a given a set of parameters.
      Input:
            - theta [K,n_a float] each column is the distribution of topics for an author
            - phi [n_dic,K float] each column is the distribution of words for a thopic
      '''
      M = np.sum(np.sum(np.log(phi.dot(theta).dot(A))*D)) # Likelihood
      ptheta = np.sum(np.sum(np.log(theta)*(alpha-1))) # Dirichlet prior 
      pPhi = np.sum(np.sum(np.log(phi)*(beta-1))) # Dirichlet prior 
#            pPhi = np.sum(np.log(np.sum(phi**self.beta,0))) # Strange prior to be checked
#            ptheta = np.sum(np.log(np.sum(theta**self.alpha,0))) # Strange prior to be checked
      return(M,ptheta,pPhi)
#%% Generate data and test
n_d = 10000
n_dic = 800
n_w = 400
n_a = 200
K = 20

beta = 0.9
alpha = 0.9
# only na = nd |
#A = np.eye(n_a)
A = np.zeros((n_a,n_d))
for d in range(n_d):
      A[np.random.choice(n_a, 4),d] = 1/4
Adic = {}
for a in  range(n_a):
      Adic[str(a)] = list(np.where(A[a,:]>0)[0])
      
      
#thetaStar = np.zeros((K,n_a))
#for a in range(n_a):
#      u = np.ones(K)
#      u[np.random.choice(K)] = alpha+1
#      thetaStar[:,a] = np.random.dirichlet(u)
#      
#      
#phiStar = np.zeros((n_dic,K))
#for a in range(K):
#      u = np.ones(n_dic)
#      u[np.random.choice(n_dic)] = beta+1
#      phiStar[:,a] = np.random.dirichlet(u)
thetaStar = np.random.dirichlet(np.ones(K)*alpha,(n_a)).transpose()
phiStar = np.random.dirichlet(np.ones(n_dic)*beta,(K)).transpose()
thetaDoc = thetaStar.dot(A)

#print(np.sum(theta,0),np.sum(varphi,0))

#----------- Doc generation
W_ = []
Z = np.zeros((n_w,n_d)).astype(int)
W = np.zeros((n_w,n_d)).astype(int)
D = np.zeros((n_dic,n_d)).astype(int)
for d in range(n_d):
      # generate number words in doc
      n_ = n_w
      # generate words
      Z[:,d] = np.random.multinomial(1,thetaDoc[:,d],n_).argmax(1).astype(int)
      for w in range(n_):
            # Store topic|
            W[w,d] = int(np.random.multinomial(1,phiStar[:,int(Z[w,d])]).argmax())
            D[W[w,d],d] += 1
      W_.append([(k,D[k,d]) for k in range(n_dic)])

#%%

t1 = time.time()
aaa = aLDA_estimator(K, W, A, alpha, beta)
aaa.gd_ll(0.0004, 120, 0,0.5983128,0,0)
#plt.plot(np.sum(aaa.llgd,1)/np.sum(aaa.loglik(thetaStar,phiStar)))
plt.plot(aaa.llgd/aaa.loglik(thetaStar,phiStar))
print('elapsed'+str(time.time()-t1))


print(np.sum(aaa.loglik(thetaStar,phiStar)),np.sum(aaa.llgd[-1]))#,aaa.loglik(theta_lda,phi_lda))
#%%
from gensim.models import AuthorTopicModel
t1 = time.time()
model = AuthorTopicModel(W_, author2doc=Adic,  num_topics=K)
print('elapsed'+str(time.time()-t1))
#%%
phiGen = model.get_topics().transpose()
thetaGen = 0*thetaStar
for a in  range(n_a):
      thetaGen[:,a] = [b for (c,b) in model.get_author_topics(str(a),0)]

#%%
n_d_test = 5000

A_test = np.zeros((n_a,n_d_test))
for d in range(n_d_test):
      A_test[np.random.choice(n_a, 4),d] = 1/4
thetaDoc_test = thetaStar.dot(A_test)
W_test_ = []
Z_test = np.zeros((n_w,n_d_test)).astype(int)
W_test = np.zeros((n_w,n_d_test)).astype(int)
D_test = np.zeros((n_dic,n_d_test)).astype(int)
for d in range(n_d_test):
      # generate number words in doc
      n_ = n_w
      # generate words
      Z_test[:,d] = np.random.multinomial(1,thetaDoc[:,d],n_).argmax(1).astype(int)
      for w in range(n_):
            # Store topic|
            W_test[w,d] = int(np.random.multinomial(1,phiStar[:,int(Z_test[w,d])]).argmax())
            D_test[W_test[w,d],d] += 1
#      W_test_.append([(k,D_test[k,d]) for k in range(n_dic)])


#%%
print('ll + Pa + Pb / Learning set')

print(loglikaLDA(thetaStar, phiStar, A, D, alpha, beta))   
print(loglikaLDA(aaa.thetaStar, aaa.phiStar, A, D, alpha, beta))        
print(loglikaLDA(thetaGen, phiGen, A, D, alpha, beta))      

#print(np.sum(loglikaLDA(thetaStar, phiStar, A, D, alpha, beta)),np.sum(loglikaLDA(aaa.thetaStar, aaa.phiStar, A, D, alpha, beta)),np.sum(loglikaLDA(thetaGen, phiGen, A, D, alpha, beta)))
print('ll + Pa + Pb / test set')
print(loglikaLDA(thetaStar, phiStar, A_test, D_test, alpha, beta))   
print(loglikaLDA(aaa.thetaStar, aaa.phiStar, A_test, D_test, alpha, beta))        
print(loglikaLDA(thetaGen, phiGen, A_test, D_test, alpha, beta))      

#print(np.sum(loglikaLDA(thetaStar, phiStar, A, D, alpha, beta)),np.sum(loglikaLDA(aaa.thetaStar, aaa.phiStar, A, D, alpha, beta)),np.sum(loglikaLDA(thetaGen, phiGen, A, D, alpha, beta)))

