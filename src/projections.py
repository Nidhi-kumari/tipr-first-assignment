import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def dolphins():
    X=pd.read_csv("../data/dolphins/dolphins.csv",encoding = 'utf8',sep='\s+',header=None)
    for r_projections in range(2,int(X.shape[1]/2 +1),2):
        print(r_projections)
        rn=np.random.standard_normal(size=(X.shape[1])*r_projections)
        rn=rn.reshape((X.shape[1]),r_projections)
        transformation_matrix=np.asmatrix(rn)
        data=np.asmatrix(X)
        X_data = data * transformation_matrix
        #print(X_data)
        filename =  '../Reduced_dim/dolphins/dolphins_%d.csv'%(r_projections,)
        np.savetxt(filename, X_data, delimiter=",")



def pubmed():
    Y=pd.read_csv("../data/pubmed/pubmed.csv",encoding = 'utf8',sep='\s+',header=None)
    for r_projections in [2,4,8,16,32,64]:
        print(r_projections)
        rn=np.random.standard_normal(size=(Y.shape[1])*r_projections)
        rn=rn.reshape((Y.shape[1]),r_projections)
        transformation_matrix=np.asmatrix(rn)
        data=np.asmatrix(Y)
        Y_data = data * transformation_matrix
        print(Y_data)
        filename =  '../Reduced_dim/pubmed/pubmed_%d.csv'%(r_projections,)
        np.savetxt(filename, Y_data, delimiter=",")
        
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
        rn=np.random.standard_normal(size=(twitter_text.shape[1])*r_projections)
        rn=rn.reshape((twitter_text.shape[1]),r_projections)
        transformation_matrix=np.asmatrix(rn)
        data=np.asmatrix(twitter_text)
        twitter_text1 = data * transformation_matrix
        #print(twitter_text1)
        filename =  '../Reduced_dim/twitter/twitter_%d.csv'%(r_projections,)
        np.savetxt(filename, twitter_text1, delimiter=",")
        
        


if __name__ == "__main__":
    print("Random Projection")
    dolphins()
    twitter()
    pubmed()
