
# coding: utf-8

# In[20]:


#!pip3 install pandas ipython[all] jupyter --user


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# In[4]:

'''
X_data=pd.read_csv("../data/dolphins/dolphins.csv",encoding = 'utf8',sep='\s+',header=None)
y=pd.read_csv("../data/dolphins/dolphins_label.csv",encoding = 'utf8',sep='\s+',names = ["label"])


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.20, random_state=42)
df_train = pd.concat([X_train, y_train],axis = 1)
df_test = pd.concat([X_test, y_test],axis = 1)
train_data = df_train.values.tolist()
test_data = df_test.values.tolist()

'''
# In[8]:


#X_train.shape


# In[9]:



def Distance(x, y):
    distance = 0
    for i in range(len(x)-1):
        distance += pow((x[i]-y[i]), 2)
    return math.sqrt(distance)


# In[10]:



def kNeighbors(data, query, k):
   distances = []
   neighbors = []
  
   for i in range(len(data)):
       distance = Distance(data[i], query)
       distances.append((data[i], distance))
   distances.sort(key=operator.itemgetter(1))
   #print(type(distances))
   #print(distances)
   for i in range(k):
       neighbors.append(distances[i][0])
   return neighbors


# In[11]:


def Predict(neighbors):
    predicted = {}
    for i in range(len(neighbors)):
        output = neighbors[i][-1]
        if output in predicted:
            predicted[output] += 1
        else:
            predicted[output] = 1
    pre = sorted(predicted.items(), key=operator.itemgetter(1), reverse=True)
    #print(pre)
    return pre[0][0]


# In[12]:


def main(train_data,test_data,y_test):
    y_pred=[]
    k = 3
    for i in range(len(test_data)):
        neighbors = kNeighbors(train_data, test_data[i], k)
        result = Predict(neighbors)
        y_pred.append(result)
    accuracy = accuracy_score(y_test, y_pred)
    f1scoremacro = f1_score(y_test, y_pred,average='macro')
    f1scoremicro = f1_score(y_test, y_pred,average='micro')
    return accuracy,f1scoremacro,f1scoremicro


# In[13]:


from sklearn.metrics import f1_score
'''
def evaluation():
    f1score=[]
    dimension=[]
    accuracies=[]
    for i in range(2,int(X_data.shape[1]/2 +1),2):
        filename =  '../Reduced_dim/dolphins/dolphins_%d.csv'%(i,)
        print(filename)
        X=pd.read_csv(filename,encoding = 'utf8',header=None)
        y=pd.read_csv("../data/dolphins/dolphins_label.csv",encoding = 'utf8',names = ["label"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        df_train = pd.concat([X_train, y_train],axis = 1)
        df_test = pd.concat([X_test, y_test],axis = 1)
        train_data = df_train.values.tolist()
        test_data = df_test.values.tolist()
        print(train_data)
        #print(int(train_data))
        accuracy,f1_scoremacro,f1_scoremicro = main(train_data,test_data,y_test)
        dimension.append(i)
        accuracies.append(accuracy)
        f1score.append(f1_score1)
    return dimension,accuracies,f1score
        


# In[14]:


evaluation()


# In[17]:


pubmed=pd.read_csv("../data/pubmed/pubmed.csv",encoding = 'utf8',sep='\s+',header=None)
from sklearn.metrics import f1_score
def evaluation_pubmed():
    f1score=[]
    dimension=[]
    accuracies=[]
    for i in range(2,int(pubmed.shape[1]/2 +1),2):
        filename =  '../Reduced_dim/pubmed/pubmed_%d.csv'%(i,)
        print(filename)
        X=pd.read_csv(filename,encoding = 'utf8',header=None)
        y=pd.read_csv("../data/pubmed/pubmed_label.csv",encoding = 'utf8',names = ["label"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        df_train = pd.concat([X_train, y_train],axis = 1)
        df_test = pd.concat([X_test, y_test],axis = 1)
        train_data = df_train.values.tolist()
        test_data = df_test.values.tolist()
        #print(train_data)
        #print(int(train_data))
        accuracy,f1_score1 = main(train_data,test_data,y_test)
        dimension.append(i)
        accuracies.append(accuracy)
        f1score.append(f1_score1)
    return dimension,accuracies,f1score

'''

def predict(train_data,train_label,test_data,test_label):
    X_train = pd.read_csv(train_data,encoding = 'utf8',sep='\s+',header=None)
    y_train = pd.read_csv(train_label,encoding = 'utf8',sep='\s+',header=None)
    X_test = pd.read_csv(test_data,encoding = 'utf8',sep='\s+',header=None)
    y_test = pd.read_csv(test_label,encoding = 'utf8',sep='\s+',header=None)
    df_train = pd.concat([X_train, y_train],axis = 1)
    df_test = pd.concat([X_test, y_test],axis = 1)
    train_data = df_train.values.tolist()
    test_data = df_test.values.tolist()
    accuracy,f1_scoremacro,f1_scoremicro = main(train_data,test_data,y_test)
    print("Nearest neighbor self created")
    print("Test accuracy :: ",accuracy)
    print("Test Macro F1-score :: ",f1_scoremacro)
    print("Test Micro F1-score :: ",f1_scoremicro)



def predict_twitter(test_data,test_label):
	
    file = "../data/twitter/twitter.txt"
    with open(file) as f:
        content = f.readlines()
    content = [str(x.strip()) for x in content] 

    vectorizer = TfidfVectorizer()
    vectorizer.fit(content)
    vector = vectorizer.transform(content)
    X = vector.toarray()

    file_y = "../data/twitter/twitter_label.txt"
    with open(file_y) as f:
        content_y = f.readlines()
    y = [str(x.strip()) for x in content_y]

    


    with open(test_data) as f:
        content_test = f.readlines()
    content_test = [str(x.strip()) for x in content_test] 
    vector = vectorizer.transform(content_test)
    X_test = vector.toarray()

    
    with open(test_label) as f:
        content_y_test = f.readlines()
    y_test = [str(x.strip()) for x in content_y_test]
    #print("reached here:")
    df_train = pd.concat([pd.DataFrame(X), pd.DataFrame(y)],axis = 1)
    df_test = pd.concat([pd.DataFrame(X_test),pd.DataFrame( y_test)],axis = 1)
    train_data = df_train.values.tolist()
    test_data = df_test.values.tolist()
    accuracy,f1_scoremacro,f1_scoremicro = main(train_data,test_data,y_test)
    print("Nearest neighbor self created")
    print("Test accuracy :: ",accuracy)
    print("Test Macro F1-score :: ",f1_scoremacro)
    print("Test Micro F1-score :: ",f1_scoremicro)




    


# In[19]:

#dimensionpubmed,accuracypubmed,f1scorepubmed = evaluation_pubmed()


# In[22]:



import pickle
#pubmedResult = "pickle.dat"
'''
f = open("pubmedResultNN", "wb")
pickle.dump(dimensionpubmed,f)
pickle.dump(accuracypubmed, f)
pickle.dump(f1scorepubmed,f)
f.close()





#with open(pubmedResult, "rb") as f:
#    print pickle.load(f)
    


# In[23]:


f = open("pubmedResultNN", "rb")
value1 = pickle.load(f)
value2 = pickle.load(f)
value3 = pickle.load(f)
f.close()


# In[24]:


value2


# In[2]:


import pickle
f = open("pubmedResultNN", "rb")
value1 = pickle.load(f)
value2 = pickle.load(f)
value3 = pickle.load(f)
f.close()


# In[3]:


value1
'''

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
