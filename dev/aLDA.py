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
from gensim.models import AuthorTopicModel,LdaModel
    
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
            pA = 1#(self.gamma - 1)*np.sum(np.log(A**self.AMask)) # Dirichlet prior constant gamma
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
                  dX = theta*(phi.T.dot(Dg.dot(A.T))-np.diag(A.dot(Dg.T.dot(phi.dot(theta)))))+Y0*(self.alpha[:, None]-1-theta*np.sum(self.alpha-1))
                  VX = b_mom*VX + (1-b_mom)*dX
                  X = X + step*VX
                  theta = normalize(np.exp(X),'l1',0)
                  
                  dY = phi*(Dg.dot(A.T.dot(theta.T))-np.diag(phi.T.dot(Dg.dot(A.T.dot(theta.T)))))+Y0*(self.beta[:, None]-1-phi*np.sum(self.beta-1))
                  VY = b_mom*VY + (1-b_mom)*dY
                  Y = Y + step*VY
                  phi = normalize(np.exp(Y),'l1',0)
                  
                  
                  dZ = A*(theta.T.dot(phi.T.dot(Dg)-np.diag(Dg.T.dot(phi.dot(theta.dot(A))))))
                  VZ = b_mom*VZ + (1-b_mom)*dZ+1*((self.gamma-1)*(1-A*np.sum(self.AMask,0)))
                  Z = Z + step*VZ    
                  A = normalize(self.AMask*np.exp(Z),'l1',0) 
                  
                  # Store ll
                  self.llgd[it+1,:] = self.loglik(theta,phi,A)
#                  if np.mod(it+1,10) ==0:
#                        print(it+1)
#                        print(self.llgd[it+1]/self.llgd[it+1-10])
                        
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
#      ptheta = np.sum(np.sum(np.log(theta)*(alpha-1))) # Dirichlet prior 
#      pPhi = np.sum(np.sum(np.log(phi)*(beta-1))) # Dirichlet prior 
#      pA = (gamma - 1)*np.sum(np.log(A**(A>0))) # Dirichlet prior constant gamma
#      return(M,ptheta,pPhi,pA)
      return(M)
      

#%% Comparison LDA/aLDA on LDA data

#%% LDA dataset A_mask = id -> n_a = n_d
n_d = 600
n_a = n_d

n_dic = 1000
n_w = 200
K = 15
beta = 3.1
alpha = 3.5
gamma = -1 
A_mask = np.eye(n_a)   

nb_fold = 10
k_list = [10,15,20,30]
nb_k = len(k_list)
aLDA_store_train = np.zeros((nb_fold,nb_k))
aLDA_store_test = np.zeros((nb_fold,nb_k))
LDA_store_train = np.zeros((nb_fold,nb_k))
LDA_store_test = np.zeros((nb_fold,nb_k)) 

aLDAgen = aLDA_generator(n_dic, n_w, A_mask, K, alpha, beta, gamma)
aLDAgen.itialise()
for f in range(nb_fold):
      train_Z,train_C,train_D,train_C_  = aLDAgen.generate()
      for k in range(nb_k):
            aLDA = aLDA_estimator(k_list[k], train_C, A_mask, 3, 3,1)
            aLDA.gd_ll(0.05, 60, 0,0.0,0,1)
            LDA = LdaModel(train_C_, num_topics=k_list[k])
            phiGen = LDA.get_topics().transpose()
            thetaGen = 0*aLDA.thetaStar
            for d in  range(n_d):
                  tmp = LDA.get_document_topics(train_C_[d])
                  ind = [c for (c,b) in tmp]
                  thetaGen[ind,d] = [b for (c,b) in tmp]
            aLDA_store_train[f,k] = aLDA.llgd[-1,0]
            LDA_store_train[f,k] = loglikaLDA(thetaGen, phiGen, A_mask, train_D,  alpha, beta,1)
            aLDA_store_test[f,k] = np.sum(np.sum(np.log(aLDA.phiStar.dot(aLDA.thetaStar).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w
            LDA_store_test[f,k] = np.sum(np.sum(np.log(phiGen.dot(thetaGen).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w
      print(f)


#plt.plot(aaa.llgd[:,0])
#print('elapsed'+str(time.time()-t1))   


#%%
max_ll = np.sum(np.sum(np.log(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w
#plt.Figure()
#plt.subplot(1,2,1)
#plt.plot(k_list,np.mean(LDA_store_train.T,1))
#plt.plot(k_list,np.mean(aLDA_store_train.T,1))
#plt.plot(k_list,LDA_store_train.T,'*')
#plt.plot(k_list,aLDA_store_train.T,'*')
#plt.title('Training set,log(p(D|theta,phi)), K= '+str(K) )
#plt.xlabel('K')
#plt.subplot(1,2,2)
plt.plot(k_list,np.mean(LDA_store_test.T,1))
plt.plot(k_list,np.mean(aLDA_store_test.T,1))
plt.plot(k_list,np.ones(nb_k)*max_ll)
plt.plot(k_list,LDA_store_test.T,'*')
plt.plot(k_list,aLDA_store_test.T,'*')
plt.title('Perplexity, K= '+str(K) )
plt.xlabel('K')
plt.legend(['LDA','aLDA','max'])


##%% Test the convergence parameters for aLDA
#k=10
#aLDA = aLDA_estimator(k, train_C, A_mask, alpha, beta,1)
#aLDA.gd_ll(0.0098, 60, 0,0.0,0,100)
#
#print(np.sum(np.sum(np.log(aLDA.phiStar.dot(aLDA.thetaStar).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w)
#a1 = aLDA.llgd
#aLDA.gd_ll(0.05, 60, 0,0.0,0,1)
#print(np.sum(np.sum(np.log(aLDA.phiStar.dot(aLDA.thetaStar).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w)
#a2 = aLDA.llgd
#plt.Figure()
#plt.subplot(1,4,1)
#plt.plot(a1[:,0])
#plt.plot(a2[:,0])
#plt.subplot(1,4,2)
#plt.plot(a1[:,1])
#plt.plot(a2[:,1])
#plt.subplot(1,4,3)
#plt.plot(a1[:,2])
#plt.plot(a2[:,2])
#plt.subplot(1,4,4)
#plt.plot(np.sum(a1,1))
#plt.plot(np.sum(a2,1))

#%% Comparison aTM/aLDA on aTM data

#%% 

n_d = 600
n_a = 400

n_dic = 1000
n_w = 200
K = 15
beta = 3.1
alpha = 3.5
gamma = -1
n_a_mean = 3

# Generate an author matrix 
A_mask = np.zeros((n_a,n_d))
for d in range(n_d):
      p =list(np.ones(n_a_mean))+list( (1+np.arange(n_a_mean))/(n_a_mean+1))[::-1]
      nb_a_ = np.random.choice(n_a_mean*2,p = p/np.sum(p),replace=False )+1
      A_mask[np.random.choice(n_a, nb_a_,replace=False),d] = 1  
Adic = {}
for a in  range(n_a):
      Adic[str(a)] = list(np.where(A_mask[a,:]>0)[0])
nb_fold = 10
k_list = [10,15,20,30]
nb_k = len(k_list)
aLDA_store_train = np.zeros((nb_fold,nb_k))
aLDA_store_test = np.zeros((nb_fold,nb_k))
aTM_store_train = np.zeros((nb_fold,nb_k))
aTM_store_test = np.zeros((nb_fold,nb_k)) 

aLDAgen = aLDA_generator(n_dic, n_w, A_mask, K, alpha, beta, gamma)
aLDAgen.itialise()
for f in range(nb_fold):
      train_Z,train_C,train_D,train_C_  = aLDAgen.generate()
      for k in range(nb_k):
            aLDA = aLDA_estimator(k_list[k], train_C, A_mask, alpha, beta,1)
            aLDA.gd_ll(0.05, 60, 0,0.0,0,1)
            aTM = AuthorTopicModel(train_C_ , author2doc=Adic, num_topics=k_list[k])
            aTM_phi = aTM.get_topics().transpose()
            aTM_theta = 0*aLDA.thetaStar
            for a in  range(n_a):
                  aTM_theta[:,a] = [b for (c,b) in aTM.get_author_topics(str(a),0)]
            aLDA_store_train[f,k] = aLDA.llgd[-1,0]
            aTM_store_train[f,k] = loglikaLDA(aTM_theta, aTM_phi, A_mask, train_D,  alpha, beta,1)
            aLDA_store_test[f,k] = np.sum(np.sum(np.log(aLDA.phiStar.dot(aLDA.thetaStar).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w
            aTM_store_test[f,k] = np.sum(np.sum(np.log(aTM_phi.dot(aTM_theta).dot(aLDAgen.A))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w
      print(f)



#%%
max_ll = np.sum(np.sum(np.log(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDA.AStar))*(aLDAgen.phi.dot(aLDAgen.theta).dot(aLDAgen.A))))*n_w
plt.Figure()
#plt.subplot(1,2,1)
#plt.plot(k_list,np.mean(aTM_store_train.T,1))
#plt.plot(k_list,np.mean(aLDA_store_train.T,1))
#plt.plot(k_list,aTM_store_train.T,'*')
#plt.plot(k_list,aLDA_store_train.T,'*')
#plt.title('Training set,log(p(D|theta,phi,A)), K= '+str(K) )
#plt.xlabel('K')
#plt.subplot(1,2,2)
plt.plot(k_list,np.mean(aTM_store_test.T,1))
plt.plot(k_list,np.mean(aLDA_store_test.T,1))
plt.plot(k_list,np.ones(nb_k)*max_ll)
plt.plot(k_list,aTM_store_test.T,'*')
plt.plot(k_list,aLDA_store_test.T,'*')
plt.title('Perplexity, K= '+str(K) )
plt.xlabel('K')
plt.legend(['aTM','aLDA','max'])
