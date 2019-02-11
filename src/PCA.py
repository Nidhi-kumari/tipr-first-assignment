import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def dolphins():
    X=pd.read_csv("../data/dolphins/dolphins.csv",encoding = 'utf8',sep='\s+',header=None)
    for r_projections in range(2,int(X.shape[1]/2 +1),2):
        scaler = StandardScaler()
        scaler.fit(X)
        data = scaler.transform(X)
        pca = PCA(n_components=r_projections)
        dataset = pca.fit_transform(data)
        filename =  '../pca_dim/dolphins/dolphins_%d.csv'%(r_projections,)
        np.savetxt(filename, dataset, delimiter=",")
        
        
def pubmed():
    Y=pd.read_csv("../data/pubmed/pubmed.csv",encoding = 'utf8',sep='\s+',header=None)
    for r_projections in range(2,int(Y.shape[1]/2 +1),2):
        scaler = StandardScaler()
        scaler.fit(Y)
        data = scaler.transform(Y)
        pca = PCA(n_components=r_projections)
        dataset = pca.fit_transform(data)
        filename =  '../pca_dim/pubmed/pubmed_%d.csv'%(r_projections,)
        np.savetxt(filename, dataset, delimiter=",")
        
        
def twitter():
    file = '../data/twitter/twitter.txt'
    with open(file) as f:
        content = f.readlines()
    content = [str(x.strip()) for x in content] 


    vectorizer = TfidfVectorizer()
    vectorizer.fit(content)
    vector = vectorizer.transform(content)
    twitter_text = vector.toarray()
    for r_projections in [2,4,8,16,32,64,128,256,512,1024]:
        scaler = StandardScaler()
        scaler.fit(twitter_text)
        data = scaler.transform(twitter_text)
        pca = PCA(n_components=r_projections)
        dataset = pca.fit_transform(data)
        filename =  '../pca_dim/twitter/twitter_%d.csv'%(r_projections,)
        np.savetxt(filename, dataset, delimiter=",")
        
        
        
if __name__ == "__main__":
    print("PCA Projection")
    dolphins()
    twitter()
    pubmed()
