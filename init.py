# -*- coding: utf-8 -*-
"""
Main module 

@author: Niraj Gautam
"""
import generate_pos
import pickle
from helper import feature_extraction
import train_models
import random
import test_models
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sn
if __name__== "__main__":
    
    generatePos = generate_pos.GeneratePos(quoteFile="quote.txt",nonQuoteFile="nonquote.txt")
    quotePos, nonQoutePos = generatePos.generate_pos()
    
    quoteFile = open ('./data/quote.txt', "rb")
    quote = pickle.load(quoteFile)
    
    nonquoteFile = open ('./data/nonquote.txt', "rb")
    nonquote = pickle.load(nonquoteFile)
        
 
    #creating word feature list from POS variables
    allowed_word_types = ["JJ","R","NN","V","VBN","VBP","VB"]
    all_words = []
    documents = []
    
   
    for q in quote:
        documents.append((q,"p"))
        for w in quotePos:
            for c in w:
                if c[1] in allowed_word_types:
                    all_words.append(c[0].lower())
               
    for n in nonquote:
        documents.append((n,"n"))
        for w in nonQoutePos:
            for c in w:
                if c[1] in allowed_word_types:
                    all_words.append(c[0].lower())
                    

    # create object of feature extraction and generate feature set
    featureExtraction= feature_extraction.Featurextraction()
    featuresets = [(featureExtraction.find_feature(rev), category) for (rev, category) in documents]

    random.shuffle(featuresets)
    print(len(featuresets))
    
    # generate train/test  dataset
    training_set = featuresets[:6000]
    testing_set  = featuresets[6000:]

    # Model training
    t= train_models.TrainModel(training_set)
    t.train()
    
    # Model testing against  testing set
    testModel= test_models.TestModel(testing_set)
    pred=testModel.test()
    
    y_test = [f[1] for f in testing_set]
    
    # plot confusion matrix for visualizing result
    cm = confusion_matrix(y_test, pred)
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    
    #log classification report 
    print(classification_report(y_test, pred))