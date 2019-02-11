# Implement code for Locality Sensitive Hashing here!
import pandas as pd
import numpy as np
from sklearn.neighbors import LSHForest
import math
import operator
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score,f1_score
import warnings
warnings.filterwarnings("ignore")


def most_common(lst):
    return max(set(lst), key=lst.count)



from sklearn.metrics import f1_score
def evaluation_pubmedLSH():
    pubmed=pd.read_csv("../data/pubmed/pubmed.csv",encoding = 'utf8',sep='\s+',header=None)
    Accuracy=dict()
    F1scoremacro=dict()
    F1scoremicro=dict()
    for file in [2,4,8,16,32]:
        filename =  '../Reduced_dim/pubmed/pubmed_%d.csv'%(file,)
        #print(filename)
        X=pd.read_csv(filename,encoding = 'utf8',header=None)
        y=pd.read_csv("../data/pubmed/pubmed_label.csv",encoding = 'utf8',names = ["label"])
        
        X = np.array(X)
        y = np.array(y)
    
        kf = KFold(y.shape[0],n_folds=10)
        lshf = LSHForest(random_state=42)
        n_neighbors=5
        
        
        accuracies=[]
        f1scoremacro=[]
        f1scoremicro=[]
        for train_index, validation_index in kf:
           #print("TRAIN:", train_index, "TEST:", test_index)
            
            #print(X)
            
            X_train, X_val = X[train_index], X[validation_index]
            y_train, y_val = y[train_index], y[validation_index]
            lshf.fit(X_train)
            distances, indices = lshf.kneighbors(X_val, n_neighbors=5)
            index = indices.tolist()
            predicted = []
            predicted_label=[]
            for i in index:
                row=[]
                for j in i:

                    y_pred = y_train[j]
                    row.append(y_pred)

                predicted.append(row)
            #print(np.asarray(predicted))
            final=np.resize(np.asarray(predicted),(X_val.shape[0],n_neighbors))
            for lst in final.tolist():
                y_val_pred = most_common(lst)
                predicted_label.append(y_val_pred)
            #print(y_val) 
            y_test=np.resize(y_val,(1,X_val.shape[0])).tolist()[0]
            #print(y_test.tolist()[0])
            #print("-----------")
            #print(predicted_label)
            accuracy = accuracy_score(y_test, predicted_label)
            f1_scoremacro = f1_score(y_test, predicted_label,average='macro')
            f1_scoremicro = f1_score(y_test, predicted_label,average='micro')
            accuracies.append(accuracy)
            f1scoremacro.append(f1_scoremacro)
            f1scoremicro.append(f1_scoremicro)
        #print(accuracies)
        #print(sum(accuracies)/float(len(accuracies)))
        Accuracy[file]=sum(accuracies)/float(len(accuracies))
        F1scoremacro[file]=sum(f1scoremacro)/float(len(f1scoremacro))
        F1scoremicro[file]=sum(f1scoremicro)/float(len(f1scoremicro))
    print("pubmed")
    
    print("Accuracy",Accuracy)
    print("F1scoremacro",F1scoremacro)
    print("F1scoremicro",F1scoremicro)
    return Accuracy,F1scoremacro,F1scoremicro





from sklearn.metrics import f1_score
def evaluation_twitterLSH():
    Accuracy=dict()
    F1scoremacro=dict()
    F1scoremicro=dict()
    
    
    file_y = "../data/twitter/twitter_label.txt"
    with open(file_y) as f:
        content_y = f.readlines()
    y = [str(x.strip()) for x in content_y]
    y = np.array(y)
    
       
    
    for file in [2,4,8,16,32,64,128,256,512,1024]:
        filename =  '../Reduced_dim/twitter/twitter_%d.csv'%(file,)
        #print(filename)
        X=pd.read_csv(filename,encoding = 'utf8',header=None)
        
    
        X = np.array(X)
        
    
        kf = KFold(y.shape[0],n_folds=10)
        lshf = LSHForest(random_state=42)
        n_neighbors=5
        
        accuracies=[]
        f1scoremacro=[]
        f1scoremicro=[]
        for train_index, validation_index in kf:
           #print("TRAIN:", train_index, "TEST:", test_index)
            
            #print(X)
            
            X_train, X_val = X[train_index], X[validation_index]
            y_train, y_val = y[train_index], y[validation_index]
            lshf.fit(X_train)
            distances, indices = lshf.kneighbors(X_val, n_neighbors=5)
            index = indices.tolist()
            predicted = []
            predicted_label=[]
            for i in index:
                row=[]
                for j in i:

                    y_pred = y_train[j]
                    row.append(y_pred)

                predicted.append(row)
            #print(np.asarray(predicted))
            final=np.resize(np.asarray(predicted),(X_val.shape[0],n_neighbors))
            for lst in final.tolist():
                y_val_pred = most_common(lst)
                predicted_label.append(y_val_pred)
            #print(y_val) 
            y_test=np.resize(y_val,(1,X_val.shape[0])).tolist()[0]
            #print(y_test.tolist()[0])
            #print("-----------")
            #print(predicted_label)
            accuracy = accuracy_score(y_test, predicted_label)
            f1_scoremacro = f1_score(y_test, predicted_label,average='macro')
            f1_scoremicro = f1_score(y_test, predicted_label,average='micro')
            accuracies.append(accuracy)
            f1scoremacro.append(f1_scoremacro)
            f1scoremicro.append(f1_scoremicro)
        #print(accuracies)
        #print(sum(accuracies)/float(len(accuracies)))
        Accuracy[file]=sum(accuracies)/float(len(accuracies))
        F1scoremacro[file]=sum(f1scoremacro)/float(len(f1scoremacro))
        F1scoremicro[file]=sum(f1scoremicro)/float(len(f1scoremicro))
    print("Twitter")
    
    print("Accuracy",Accuracy)
    print("F1scoremacro",F1scoremacro)
    print("F1scoremicro",F1scoremicro)
    return Accuracy,F1scoremacro,F1scoremicro




from sklearn.metrics import f1_score
def evaluation_dolphinsLSH():
    Accuracy=dict()
    F1scoremacro=dict()
    F1scoremicro=dict()
    for file in [2,4,8,16]:
        filename =  '../Reduced_dim/dolphins/dolphins_%d.csv'%(file,)
        #print(filename)
        X=pd.read_csv(filename,encoding = 'utf8',header=None)
        y=pd.read_csv("../data/dolphins/dolphins_label.csv",encoding = 'utf8',names = ["label"])
        
        X = np.array(X)
        y = np.array(y)
    
        kf = KFold(y.shape[0],n_folds=10)
        lshf = LSHForest(random_state=42)
        n_neighbors=5
        
        
        accuracies=[]
        f1scoremacro=[]
        f1scoremicro=[]
        for train_index, validation_index in kf:
           #print("TRAIN:", train_index, "TEST:", test_index)
            
            #print(X)
            
            X_train, X_val = X[train_index], X[validation_index]
            y_train, y_val = y[train_index], y[validation_index]
            lshf.fit(X_train)
            distances, indices = lshf.kneighbors(X_val, n_neighbors=5)
            index = indices.tolist()
            predicted = []
            predicted_label=[]
            for i in index:
                row=[]
                for j in i:

                    y_pred = y_train[j]
                    row.append(y_pred)

                predicted.append(row)
            #print(np.asarray(predicted))
            final=np.resize(np.asarray(predicted),(X_val.shape[0],n_neighbors))
            for lst in final.tolist():
                y_val_pred = most_common(lst)
                predicted_label.append(y_val_pred)
            #print(y_val) 
            y_test=np.resize(y_val,(1,X_val.shape[0])).tolist()[0]
            #print(y_test.tolist()[0])
            #print("-----------")
            #print(predicted_label)
            accuracy = accuracy_score(y_test, predicted_label)
            f1_scoremacro = f1_score(y_test, predicted_label,average='macro')
            f1_scoremicro = f1_score(y_test, predicted_label,average='micro')
            accuracies.append(accuracy)
            f1scoremacro.append(f1_scoremacro)
            f1scoremicro.append(f1_scoremicro)
        #print(accuracies)
        #print(sum(accuracies)/float(len(accuracies)))
        Accuracy[file]=sum(accuracies)/float(len(accuracies))
        F1scoremacro[file]=sum(f1scoremacro)/float(len(f1scoremacro))
        F1scoremicro[file]=sum(f1scoremicro)/float(len(f1scoremicro))
    print("dolphins")
    print("Accuracy",Accuracy)
    print("F1scoremacro",F1scoremacro)
    print("F1scoremicro",F1scoremicro)
    return Accuracy,F1scoremacro,F1scoremicro




def run(data):
   print("LSH")
   if data == "dolphins":
       evaluation_dolphinsLSH()
   elif data == "twitter":
       evaluation_twitterLSH()
   elif data == "pubmed":
       evaluation_pubmedLSH()

import argparse
from sys import argv

if __name__ == "__main__":
    evaluation_dolphinsLSH()
    
    evaluation_pubmedLSH()
    evaluation_twitterLSH()


