import run
import nn
import bayes
import MNB
import argparse
#import lsh
from sys import argv
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
   print('Welcome to the world of high and low dimensions!')
   # The entire code should be able to run from this file!
   test_data= argv[2]
   test_label = argv[4]
   data = argv[6]
   if data == "dolphins":
       run.predict_dolphins(test_data,test_label)
       nn.predict("../data/dolphins/dolphins.csv","../data/dolphins/dolphins_label.csv",test_data,test_label)
       bayes.predictNB("../data/dolphins/dolphins.csv","../data/dolphins/dolphins_label.csv",test_data,test_label)
       #lsh.run("dolphins")
   elif data == "twitter":
       MNB.predict(test_data,test_label)
       #run.predict_twitter(test_data,test_label)
       #lsh.run("twitter")
       nn.predict_twitter(test_data,test_label)
       
       
   elif data == "pubmed":
       
       bayes.predictNB("../data/pubmed/pubmed.csv","../data/pubmed/pubmed_label.csv",test_data,test_label)
       #lsh.run("pubmed")
       run.predict_pubmed(test_data,test_label)
       nn.predict("../data/pubmed/pubmed.csv","../data/pubmed/pubmed_label.csv",test_data,test_label)
       
