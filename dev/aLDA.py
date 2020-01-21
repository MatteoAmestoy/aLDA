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

class aLDA_estimator():
      def __init__(self, K, data, AMask, alpha, beta, gamma):
            '''
            Input:
            - K = [int] nb of topics
            - data = [n_w,n_d int] nb words * nb doc matrix of word index
            - AMask = [n_a,n_d 0-1] nb authors * nb doc matrix of author participation to each paper
                  (1 if author participated to paper)
            - alpha[n_a float] priors on theta
            - beta [n_dic float] priors on  phi
            - gamma [flaot] prior on A
            '''
            self.K = K # [int] nb of topics
            self.AMask = AMask # [n_a,n_d float] matrix of author participation to each paper (1 if author participated to paper)
            self.n_a,self.n_d = self.AMask.shape # [int] nb authors 
            self.M = data # [n_w,n_d int] matrix of word index
            self.n_dic = int(data.max())+1 # [int] nb words in dictionary
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
            
            self.D = np.zeros((self.n_dic,self.n_d)) # [n_dic,n_d int] matrix of count for each word 
            for w in range(self.n_dic):
                  self.D[w,:] = np.sum(self.M==w,0) 
            
                        
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
            pA = (self.gamma - 1)*np.sum(np.log(A**self.AMask)) # Dirichlet prior constant gamma
            return(M,pTheta,pPhi,pA)
            
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
            self.llgd = np.zeros((n_itMax,4))
            
            # initialize ------------------------------------------------------
            X = np.random.normal(0,1,(self.K,self.n_a))
            Y = np.random.normal(0,1,(self.n_dic,self.K))
            Z = np.random.normal(0,1,(self.AMask.shape))#np.ones((self.AMask.shape))#Î
            
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
                  dX = theta*(phi.T.dot(Dg.dot(A.T))-np.diag(A.dot(Dg.T.dot(phi.dot(theta)))))+1*(self.alpha[:, None]-1-theta*np.sum(self.alpha-1))
                  VX = b_mom*VX + (1-b_mom)*dX
                  X = X + step*VX
                  theta = normalize(np.exp(X),'l1',0)
                  
                  dY = phi*(Dg.dot(A.T.dot(theta.T))-np.diag(phi.T.dot(Dg.dot(A.T.dot(theta.T)))))+1*(self.beta[:, None]-1-phi*np.sum(self.beta-1))
                  VY = b_mom*VY + (1-b_mom)*dY
                  Y = Y + step*VY
                  phi = normalize(np.exp(Y),'l1',0)
                  
                  
                  dZ = A*(theta.T.dot(phi.T.dot(Dg)-np.diag(Dg.T.dot(phi.dot(theta.dot(A))))))
                  VZ = b_mom*VZ + (1-b_mom)*dZ+1*((self.gamma-1)*(1-A*np.sum(self.AMask,0)))
                  Z = Z + step*VZ    
                  A = normalize(self.AMask*np.exp(Z),'l1',0) 
                  
                  # Store ll
                  self.llgd[it+1,:] = self.loglik(theta,phi,A)
                  if np.mod(it+1,10) ==0:
                        print(it+1)
                        print(self.llgd[it+1]/self.llgd[it+1-10])
                        
            # Store estimates
            self.thetaStar = theta
            self.phiStar = phi
            self.AStar = A

def loglikaLDA(theta, phi, A, D, alpha, beta, gamma):
      '''
      Computes the log likelihood and priors of a given a set of parameters.
      Input:
            - theta [K,n_a float] each column is the distribution of topics for an author
            - phi [n_dic,K float] each column is the distribution of words for a thopic
      '''
      M = np.sum(np.sum(np.log(phi.dot(theta).dot(A))*D)) # Likelihood
      ptheta = np.sum(np.sum(np.log(theta)*(alpha-1))) # Dirichlet prior 
      pPhi = np.sum(np.sum(np.log(phi)*(beta-1))) # Dirichlet prior 
      pA = (gamma - 1)*np.sum(np.log(A**(A>0))) # Dirichlet prior constant gamma
      return(M,ptheta,pPhi,pA)
      
      
#%% Data generation 
      
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
            - gamma [flaot] prior on A
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
            self.theta = np.random.dirichlet(self.alpha,(n_a)).transpose()
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
                  n_ = n_w
                  # generate words
                  Z[:,d] = np.random.multinomial(1,self.thetaDoc[:,d],n_).argmax(1).astype(int)
                  for w in range(n_):
                        # Store topic|
                        C[w,d] = int(np.random.multinomial(1,self.phi[:,int(Z[w,d])]).argmax())
                        D[C[w,d],d] += 1
                  C_.append([(k,D[k,d]) for k in range(self.n_dic)])
            return(Z,C,D,C_)

        
#%% Generate data and test
n_d = 30
n_dic = 200
n_w = 20
n_a = 10
n_a_mean = 1
K = 60

beta = 1.1
alpha = 1.5
gamma = 2
# only na = nd |
#A = np.eye(n_a)
AStar = np.zeros((n_a,n_d))
Ddic = {}
for d in range(n_d):
      p =list(np.ones(n_a_mean))+list( (1+np.arange(n_a_mean))/(n_a_mean+1))[::-1]
      nb_a_ = np.random.choice(n_a_mean*2,p = p/np.sum(p),replace=False )+1
      AStar[np.random.choice(n_a, nb_a_,replace=False),d] = 1/nb_a_#np.random.dirichlet(np.ones(nb_a_)*gamma)#
      Ddic[str(d)] =  [str(a) for a in list(np.where(AStar[:,d]>0)[0])]
A_mask = AStar>0
Adic = {}

for a in  range(n_a):
      Adic[str(a)] = list(np.where(AStar[a,:]>0)[0])
ddd = aLDA_generator(n_dic, n_w, A_mask, K, 1, 1, 113.3)

ddd.itialise()
Z,C,D,C_ = ddd.generate()
print(Z)    

#%%

t1 = time.time()
aaa = aLDA_estimator(K, W, AMask, alpha, beta,gamma)
aaa.gd_ll(0.0151, 150, 0,0.0,0,0)

plt.plot(aaa.llgd/aaa.loglik(thetaStar,phiStar,AStar))
print('elapsed'+str(time.time()-t1))



#print(np.sum(aaa.loglik(thetaStar,phiStar,AStar)),np.sum(aaa.llgd[-1]))#,aaa.loglik(theta_lda,phi_lda))
#print(aaa.AStar[:,1])

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
n_d_test = n_d
A_test = AStar
#A_test = np.zeros((n_a,n_d_test))
#for d in range(n_d_test):
#      A_test[np.random.choice(n_a, 4),d] = 1/4
thetaDoc_test = thetaStar.dot(A_test)
#thetaDoc_test = thetaDoc
W_test_ = []
Z_test = np.zeros((n_w,n_d_test)).astype(int)
W_test = np.zeros((n_w,n_d_test)).astype(int)
D_test = np.zeros((n_dic,n_d_test)).astype(int)
for d in range(n_d_test):
      # generate number words in doc
      n_ = n_w
      # generate words
      Z_test[:,d] = np.random.multinomial(1,thetaDoc_test[:,d],n_).argmax(1).astype(int)
      for w in range(n_):
            # Store topic|
            W_test[w,d] = int(np.random.multinomial(1,phiStar[:,int(Z_test[w,d])]).argmax())
            D_test[W_test[w,d],d] += 1


#%%
print('ll + Pa + Pb / Learning set')

print(loglikaLDA(thetaStar, phiStar, AStar, D, alpha, beta, gamma))   
print(loglikaLDA(aaa.thetaStar, aaa.phiStar, aaa.AStar, D, alpha, beta, gamma))        
print(loglikaLDA(thetaGen, phiGen, AMask/np.sum(AMask,0), D, alpha, beta, gamma))      

print('ll + Pa + Pb / test set')
print(loglikaLDA(thetaStar, phiStar, A_test, D_test, alpha, beta, gamma))   
print(loglikaLDA(aaa.thetaStar, aaa.phiStar, aaa.AStar, D_test, alpha, beta, gamma))        
print(loglikaLDA(thetaGen, phiGen, AMask/np.sum(AMask,0), D_test, alpha, beta, gamma))      

#%%

np.sum(np.abs(AMask/np.sum(AMask,0)-AStar))



