# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:24:23 2020

@author: admin
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from gensim.models import AuthorTopicModel,LdaModel
import time
#%%
x = loadmat(r'C:\Users\admin\Documents\Data_NIPS\nips_1-17.mat')
#%%

M = x['counts']
A = x['docs_authors']

At = np.asarray(A.todense().T)
M_full = np.asarray(M.todense())
empty_idx = np.where(np.sum(M_full,0)==0)
M_full = np.delete(M_full,empty_idx,1)
At = np.delete(At,empty_idx,1)
n_dic,n_doc = M_full.shape
n_a = At.shape[0]
#%% Split train test
test_pct = 20
M_test = M_full*0

for i in range(n_doc):
      tmp = sum([[j]*M_full[j,i] for j in range(n_dic)],[])
      unique, counts = np.unique(np.random.choice(tmp,int(test_pct*len(tmp)/100),False), return_counts=True)
      M_test[unique,i] = counts

M_train = M_full-M_test
#%% LDA ---------------------------------------
# data preparation
train_C_ = []
for d in range(n_doc):
      train_C_.append([(k,M_train[k,d]) for k in range(n_dic)])

#%%
l_k = [10,50,100,150]
n_k = len(l_k)      
store = np.zeros((n_k,5))
      


#%%
i = 0
for K in l_k:
      t = time.time()
      LDA = LdaModel(train_C_, num_topics=K)
      phiLDA = LDA.get_topics().transpose()
      thetaLDA = np.zeros((K,n_doc))
      for d in  range(n_doc):
            tmp = LDA.get_document_topics(train_C_[d])
            ind = [c for (c,b) in tmp]
            thetaLDA[ind,d] = [b for (c,b) in tmp]
      t1 = time.time()
      print('LDA for k = '+str(K), ', time = ' +str(t1-t) )
      t = t1
      # aLDA on LDA      
      init = {}
      init['A'] = np.eye(n_doc)
      init['theta'] = thetaLDA
      init['phi'] = phiLDA
      
      aLDALDA = aLDA_estimator(K, M_train, np.eye(n_doc), 5, 5, 1, True,init)
      aLDALDA.gd_ll(0.00004, 60, 0,0,0,1,0)
      plt.plot(aLDALDA.llgd[:,0]/aLDALDA.llgd[0,0])
      t1 = time.time()
      print('gd on LDA for k = '+str(K), ', time = ' +str(t1-t) )
      t = t1

      # aTM ---------------------------------------------------

      Adic = {}
      for a in  range(n_a):
            Adic[str(a)] = list(np.where(At[a,:]>0)[0])
            
      aTM = AuthorTopicModel(train_C_ , author2doc=Adic, num_topics=K)
      phiaTM = aTM.get_topics().transpose()
      thetaaTM = np.zeros((K,n_a))
      for a in  range(n_a):
            thetaaTM[:,a] = [b for (c,b) in aTM.get_author_topics(str(a),0)]
      t1 = time.time()
      print('aTM for k = '+str(K), ', time = ' +str(t1-t) )
      t = t1
      
      # aLDA on aTM
      init2 = {}
      init2['A'] = normalize(At,'l1',0)
      init2['theta'] = thetaaTM
      init2['phi'] = phiaTM
      
      aLDATaTM = aLDA_estimator(K, M_train, At, 10, 10, 1, True,init2)
      aLDATaTM.gd_ll(0.00004, 60, 0,0.0,0,1,0)
      plt.plot(aLDATaTM.llgd[:,0]/aLDATaTM.llgd[0,0])
      t1 = time.time()
      print('gd on aTM for k = '+str(K), ', time = '+ str(t1-t) )
      t = t1
      
      
      aLDA = aLDA_estimator(K, M_train, At, 10, 10, 1, True,init2)
      aLDA.gd_ll(0.00004, 60, 0,0.0,0,1,0.00004)
      plt.plot(aLDA.llgd[:,0]/aLDA.llgd[0,0])
      t1 = time.time()
      print('aLDA for k = '+str(K))
      store[i,0] = loglikaLDA(thetaLDA, phiLDA, np.eye(n_doc), M_test,  0, 0,1)
      store[i,1] = loglikaLDA(aLDALDA.thetaStar, aLDALDA.phiStar, aLDALDA.AStar, M_test,  0, 0,1)
      store[i,2] = loglikaLDA(thetaaTM, phiaTM, normalize(At,'l1',0), M_test,  0, 0,1)
      store[i,3] = loglikaLDA(aLDATaTM.thetaStar, aLDATaTM.phiStar, aLDATaTM.AStar, M_test,  0, 0,1)
      store[i,4] = loglikaLDA(aLDA.thetaStar, aLDA.phiStar, aLDA.AStar, M_test,  0, 0,1)
      print('End for k ='+str(K))
      i += 1
#%%
print('Train')
print(loglikaLDA(thetaLDA, phiLDA, np.eye(n_doc), M_train,  0, 0,1))
print(loglikaLDA(aLDA.thetaStar, aLDA.phiStar, aLDA.AStar, M_train, 0, 0,1))
print(loglikaLDA(thetaaTM, phiaTM, normalize(At,'l1',0), M_train, 0, 0,1))
print(loglikaLDA(aLDATaTM.thetaStar, aLDATaTM.phiStar, aLDATaTM.AStar, M_train, 0, 0,1))

print('Test')
print(loglikaLDA(thetaLDA, phiLDA, np.eye(n_doc), M_test,  0, 0,1))
print(loglikaLDA(aLDA.thetaStar, aLDA.phiStar, aLDA.AStar, M_test,  0, 0,1))
print(loglikaLDA(thetaaTM, phiaTM, normalize(At,'l1',0), M_test,  0, 0,1))
print(loglikaLDA(aLDATaTM.thetaStar, aLDATaTM.phiStar, aLDATaTM.AStar, M_test,  0, 0,1))

#%%

plt.plot(l_k,store)
plt.legend(['LDA', 'gd LDA', 'aTM', 'gd aTM', 'aLDA' ])
plt.title('Perplexity evaluated on 20% dropout')