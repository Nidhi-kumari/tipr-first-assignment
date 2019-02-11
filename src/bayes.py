#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# In[2]:




def separateByClass(dataset):
    separated = {}
    for index, row in dataset.iterrows():
        vector = row
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(list(vector))
    return separated


# In[8]:


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated







# In[11]:


import math
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)





def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries






def summarizeByClass(dataset):
   separated = separateByClass(dataset)
   summaries = {}
   for classValue, instances in separated.items():
       summaries[classValue] = summarize(instances)
   return summaries






# In[20]:


import math
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# In[21]:


def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities


# In[43]:


def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


# In[44]:


def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions


# In[48]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


'''
def main():
    y_test=[]
    X_train, X_test = train_test_split(dataset, test_size=0.20, random_state=42)
    for i in range(len(X_test)):
        y_test.append(X_test[i][-1])
    summaries = summarizeByClass(X_train)
    # test model
    predictions = getPredictions(summaries, X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1score = f1_score(y_test, predictions,average='macro')
    return accuracy,f1score


pubmed=pd.read_csv("../data/pubmed/pubmed.csv",encoding = 'utf8',sep='\s+',header=None)
from sklearn.metrics import f1_score
def evaluation_pubmed():
    f1score=[]
    dimension=[]
    accuracies=[]
    
    for i in range(2,int(pubmed.shape[1]/2 +1),2):
        y_test=[]
        filename =  '../Reduced_dim/pubmed/pubmed_%d.csv'%(i,)
        dimension.append(i)
        print(filename)
        X = pd.read_csv(filename,encoding = 'utf8',header=None)
        y = pd.read_csv("../data/pubmed/pubmed_label.csv",encoding = 'utf8',names = ["label"])
        df = pd.concat([X, y],axis =1)
        dataset=df.values.tolist()
        X_train, X_test = train_test_split(dataset, test_size=0.20, random_state=42)
        for i in range(len(X_test)):
            y_test.append(X_test[i][-1])
        summaries = summarizeByClass(X_train)
        #print(len(X_test))
        #print(len(y_test))
        predictions = getPredictions(summaries, X_test)
        #print(len(predictions))
        accuracy = accuracy_score(y_test, predictions)
        f1_score1 = f1_score(y_test, predictions,average='macro')
        
        accuracies.append(accuracy)
        f1score.append(f1_score1)
    return dimension,accuracies,f1score
        

dimensionnb,accuracynb,f1scorenb = evaluation_pubmed()



'''



def predictNB(train_data,train_label,test_data,test_label):
    X_train = pd.read_csv(train_data,encoding = 'utf8',sep='\s+',header=None)
    y_train = pd.read_csv(train_label,encoding = 'utf8',sep='\s+',header=None)
    X_test = pd.read_csv(test_data,encoding = 'utf8',sep='\s+',header=None)
    y_test = pd.read_csv(test_label,encoding = 'utf8',sep='\s+',header=None)
    df_train = pd.concat([X_train, y_train],axis = 1)
    df_test = pd.concat([X_test, y_test],axis = 1)
    X_train=df_train.values.tolist()
    X_test=df_test.values.tolist()
    y_test=[]
    for i in range(len(X_test)):
            y_test.append(X_test[i][-1])
    summaries = summarizeByClass(X_train)
    #print(len(X_test))
    #print(len(y_test))
    predictions = getPredictions(summaries, X_test)
    #print(len(predictions))
    accuracy = accuracy_score(y_test, predictions)
    f1_scoremacro = f1_score(y_test, predictions,average='macro')
    f1_scoremicro = f1_score(y_test, predictions,average='micro')
    print("Naive bayes self created")
    print("Test accuracy :: ",accuracy)
    print("Test Macro F1-score :: ",f1_scoremacro)
    print("Test Micro F1-score :: ",f1_scoremicro)






def predictNB_twitter(test_data,test_label):
	
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
    y = [int(x.strip()) for x in content_y]

    


    with open(test_data) as f:
        content_test = f.readlines()
    content_test = [str(x.strip()) for x in content_test] 
    vector = vectorizer.transform(content_test)
    X_test = vector.toarray()

    
    with open(test_label) as f:
        content_y_test = f.readlines()
    y_test = [int(x.strip()) for x in content_y_test]

    df_train = pd.concat([pd.DataFrame(X), pd.DataFrame(y)],axis = 1)
    df_test = pd.concat([pd.DataFrame(X_test),pd.DataFrame( y_test)],axis = 1)
    X_train=df_train.values.tolist()
    X_test=df_test.values.tolist()
    y_test=[]
    for i in range(len(X_test)):
            y_test.append(X_test[i][-1])
    summaries = summarizeByClass(X_train)
    #print(len(X_test))
    #print(len(y_test))
    predictions = getPredictions(summaries, X_test)
    #print(len(predictions))
    accuracy = accuracy_score(y_test, predictions)
    f1_scoremacro = f1_score(y_test, predictions,average='macro')
    f1_scoremicro = f1_score(y_test, predictions,average='micro')
    print("Naive bayes self created")
    print("Test accuracy :: ",accuracy)
    print("Test Macro F1-score :: ",f1_scoremacro)
    print("Test Micro F1-score :: ",f1_scoremicro)





# In[68]:





# In[2]:


import pickle
#pubmedResult = "pickle.dat"

'''
f = open("pubmedResultNB", "wb")
pickle.dump(dimensionnb,f)
pickle.dump(accuracynb, f)
pickle.dump(f1scorenb,f)
f.close()

'''


# In[3]:
'''

f = open("pubmedResultNB", "rb")
value1 = pickle.load(f)
value2 = pickle.load(f)
value3 = pickle.load(f)
f.close()

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



# In[ ]:




