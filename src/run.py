import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# In[3]:

import warnings
warnings.filterwarnings("ignore")


def predict_dolphins(filepath1,filepath2):
   X_train = pd.read_csv("../data/dolphins/dolphins.csv",encoding = 'utf8',sep='\s+',header=None)
   y_train = pd.read_csv("../data/dolphins/dolphins_label.csv",encoding = 'utf8',sep='\s+',header=None)
   X_test = pd.read_csv(filepath1,encoding = 'utf8',sep='\s+',header=None)
   y_test = pd.read_csv(filepath2,encoding = 'utf8',sep='\s+',header=None)
   neigh = KNeighborsClassifier(n_neighbors=3)
   neigh.fit(X_train, y_train) 
   y_predkn=neigh.predict(X_test)
   accuracykn = accuracy_score(y_test, y_predkn)
   f1scorekn = f1_score(y_test, y_predkn,average='macro')
   f1scoreknmicr = f1_score(y_test, y_predkn,average='micro')
   print()
   print("Nearest neighbor from library")
   print("Test accuracy :: ",accuracykn)
   print("Test Macro F1-score :: ",f1scorekn)
   print("Test Micro F1-score :: ",f1scoreknmicr)


def predict_pubmed(filepath1,filepath2):
   X_train = pd.read_csv("../data/pubmed/pubmed.csv",encoding = 'utf8',sep='\s+',header=None)
   y_train = pd.read_csv("../data/pubmed/pubmed_label.csv",encoding = 'utf8',sep='\s+',header=None)
   X_test = pd.read_csv(filepath1,encoding = 'utf8',sep='\s+',header=None)
   y_test = pd.read_csv(filepath2,encoding = 'utf8',sep='\s+',header=None)
   neigh = KNeighborsClassifier(n_neighbors=3)
   neigh.fit(X_train, y_train) 
   y_predkn=neigh.predict(X_test)
   accuracykn = accuracy_score(y_test, y_predkn)
   f1scorekn = f1_score(y_test, y_predkn,average='macro')
   f1scoreknmicr = f1_score(y_test, y_predkn,average='micro')
   print()
   print("Nearest neighbor from library")
   print("Test accuracy :: ",accuracykn)
   print("Test Macro F1-score :: ",f1scorekn)
   print("Test Micro F1-score :: ",f1scoreknmicr)

'''
def predict_pubmed(filepath1,filepath2):
   X_train = pd.read_csv("../data/twitter/twitter.csv",encoding = 'utf8',sep='\s+',header=None)
   y_train = pd.read_csv("../data/twitter/twitter_label.csv",encoding = 'utf8',sep='\s+',header=None)
   X_test = pd.read_csv(filepath1,encoding = 'utf8',sep='\s+',header=None)
   y_test = pd.read_csv(filepath2,encoding = 'utf8',sep='\s+',header=None)
   neigh = KNeighborsClassifier(n_neighbors=3)
   neigh.fit(X_train, y_train) 
   y_predkn=neigh.predict(X_test)
   accuracykn = accuracy_score(y_test, y_predkn)
   f1scorekn = f1_score(y_test, y_predkn,average='macro')
   f1scoreknmicr = f1_score(y_test, y_predkn,average='micro')
   print()
   print("Nearest neighbor from library")
   print("Test accuracy :: ",accuracykn)
   print("Test Macro F1-score :: ",f1scorekn)
   print("Test Micro F1-score :: ",f1scoreknmicr)
  
   
'''   


# In[4]:


import argparse
from sys import argv


if __name__ == "__main__":
   
   #parser = argparse.ArgumentParser()
   #parser.add_argument('--test-file', type=str )
   #parser.add_argument('--test-label', type=str)
   #parser.add_argument('--dataset', type=str)
   #args = parser.parse_args()
   
   test_data= argv[2]
   test_label = argv[4]
   data = argv[6]
   if data == "dolphins":
       predict_dolphins(test_data,test_label)
   elif data == "twitter":
       predict_twitter(test_data,test_label)
   elif data == "pubmed":
       predict_pubmed(test_data,test_label)

