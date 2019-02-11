#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
import math


# In[5]:

def openfile():
	file = "../data/twitter/twitter.txt"
	with open(file) as f:
	    content = f.readlines()
	content = [str(x.strip()) for x in content] 


	# In[19]:


	file_y = "../data/twitter/twitter_label.txt"
	with open(file_y) as f:
	    content_y = f.readlines()
	y = [str(x.strip()) for x in content_y]
	return content,y

# In[23]:





# In[29]:

def createDOC(content,y):
	document_list=[]
	for i in range(len(content)):
	    inner_list=[]
	    inner_list.append(str(y[i]))
	    inner_list.append(content[i])
	    document_list.append(inner_list)
	return document_list

# In[97]:


#document_list


# In[31]:


count_class={}
def count_classes(document_list):
    
    class1=['1','0','-1']
    for clas in class1:
        count=0
        for document in document_list:
            if clas==document[0]:
                count+=1

        count_class[clas]=count
             


# In[32]:


#count_classes(document_list)


# In[33]:


count_class


# In[57]:


import nltk

def getVocab(content):
    vocab=set()
    for doc in content:
        tokenized_text = nltk.word_tokenize(doc)
        for i in tokenized_text:
            vocab.add(i)
    return vocab


# In[98]:


#vocab1= sorted(list(getVocab(content)),key=str.lower)
#print(vocab1)


# In[34]:


#print(sum(count_class.values()))


# In[35]:


class_probability={}
def class_probabilities(class1,countclass):
    total=sum(countclass.values())
    for clas in class1:
        p=countclass[clas]/total
        class_probability[clas]=p
        


# In[36]:


#class_probabilities(class1,count_class)





def mergedDocument(document_list):
    merged_document={}
    class1=['1','0','-1']
    for clas in class1:
        string=""
        for l in document_list:
            if l[0]==clas:
                string+=l[1]

        merged_document[clas]=string   
    return merged_document


# In[66]:


#mergedDocument(document_list)


# In[42]:


#merged_document=mergedDocument(document_list)


# In[70]:


#print(merged_document['1'])


# In[61]:



    


# In[71]:


#documentwithwords


# In[63]:


def likelihood_prob(vocab,documentswithwords):
    class1=['1','0','-1']
    probabilities = {}
    for word in vocab: #take each word from vocab
        for clas  in class1:
            li=documentswithwords[clas]
            ntext=li.count(word)
            probabilities[(word,clas)] = (ntext+1)/(len(li)+len(vocab))
            #print(probabilities[(word,clas)])
    return probabilities


# In[64]:


#liklihood_probabiliteies=likelihood_prob(vocab1,documentwithwords)


# In[65]:


#print(liklihood_probabiliteies)


# In[88]:


def prob_classgivendoc(document_list2,documentwithwords,liklihood_probabiliteies,vocab1):
    predicted_classes=list()
    class1=['1','0','-1']
    for li in document_list2:
        dict1={}
        
        for clas in class1:
            prob=1
            for test_word in set(nltk.word_tokenize(li[1])):
                if (test_word,clas) in liklihood_probabiliteies.keys():
                    prob=prob*liklihood_probabiliteies[(test_word,clas)]
                else:
                    prob=prob*(1/(len(documentwithwords[clas])+len(vocab1)))
            dict1[clas] = prob*class_probability[clas]
            
                               
                               
        predicted_class=max(dict1,key=dict1.get)   
        predicted_classes.append(predicted_class) 
    #print(predicted_classes)    
    return predicted_classes   
                               
                               


# In[95]:


from sklearn.metrics import f1_score
def predict(X_test,y_test):
    class1=['1','0','-1']
    content,y=openfile()
    document_list=createDOC(content,y)
    
    count_classes(document_list)
    vocab1= sorted(list(getVocab(content)),key=str.lower)
    class_probabilities(class1,count_class)
    merged_document=mergedDocument(document_list)
    documentwithwords={}
    for i in merged_document.keys():
        lis= nltk.word_tokenize(merged_document[i])
        documentwithwords[i]=lis
    

    liklihood_probabiliteies=likelihood_prob(vocab1,documentwithwords)	
    file = X_test
    with open(file) as f:
        content = f.readlines()
    content = [str(x.strip()) for x in content] 
    
    file_y = y_test
    with open(file_y) as f:
        content_y = f.readlines()
    y = [str(x.strip()) for x in content_y]
    
    document_list=[]
    for i in range(len(content)):
        inner_list=[]
        inner_list.append(str(y[i]))
        inner_list.append(content[i])
        document_list.append(inner_list)
    predicted_classes=prob_classgivendoc(document_list,documentwithwords,liklihood_probabiliteies,vocab1)
    accuracy=accuracy_score(y,predicted_classes)
    f1_scoremacro = f1_score(y,predicted_classes,average='macro')
    f1_scoremicro = f1_score(y,predicted_classes,average='micro')
    print("MULTINOMIAL Naive bayes self created")
    print("Test accuracy :: ",accuracy)
    print("Test Macro F1-score :: ",f1_scoremacro)
    print("Test Micro F1-score :: ",f1_scoremicro)
    return predicted_classes


# In[ ]:


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


# In[96]:





# In[ ]:




# In[ ]:





# In[231]:





# In[ ]:




