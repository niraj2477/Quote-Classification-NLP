# -*- coding: utf-8 -*-
"""
@author: Niraj Gautam
"""

import model
import ensemble
from helper import feature_extraction

def test(featureExtraction,classifier,doc):
    feats = featureExtraction.find_feature(doc)
    return classifier.classify(feats), classifier.confidence(feats)
    
if __name__== "__main__":

    featureExtraction= feature_extraction.Featurextraction()
    
    m=model.Model()
    ONB_Clf = m.load_model('./models/ONB_clf.pickle')
    MNB_Clf = m.load_model('./models/MNB_clf.pickle')
    BNB_Clf = m.load_model('./models/BNB_clf.pickle')
    LogReg_Clf = m.load_model('./models/LogReg_clf.pickle')
    SGD_Clf = m.load_model('./models/SGD_clf.pickle')
    SVC_clf = m.load_model('./models/SVC_clf.pickle')
    
    ensemble_clf = ensemble.EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf,SVC_clf)
    
    doc=[
        "Her beauty is laced in her strength and interwoven through her flaws. She embodies perfection.",
        " what is python?",
        "I found it to be because of only 1's or 0's wound up in my y_test since my sample size was really small. ",
        "Positive anything is better than negative nothing.",
        "Be positive. Be true. Be kind.",
        "Write it on your heart that every day is the best day in the year."
        ]
    result=[]
    for i in doc:
        output=test(featureExtraction,ensemble_clf,i)
        d=[i,output]
        result.append(d)
  
    print(result)
        
        