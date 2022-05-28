# -*- coding: utf-8 -*-
"""
Created on Fri May 27 08:40:36 2022

@author: Admin
"""
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
import pickle
class TrainModel:
    
    def __init__(self,training_set):
        self.training_set=training_set
        
    
    def train(self):
        classifier = nltk.NaiveBayesClassifier.train(self.training_set)
        pickle.dump(classifier, open('./models/ONB_clf.pickle', 'wb'))
        
        MNB_clf = SklearnClassifier(MultinomialNB())
        MNB_clf.train(self.training_set)
        pickle.dump(MNB_clf, open('./models/MNB_clf.pickle', 'wb'))
        
        BNB_clf = SklearnClassifier(BernoulliNB())
        BNB_clf.train(self.training_set)
        pickle.dump(BNB_clf, open('./models/BNB_clf.pickle', 'wb'))
        
        LogReg_clf = SklearnClassifier(LogisticRegression())
        LogReg_clf.train(self.training_set)
        pickle.dump(LogReg_clf, open('./models/LogReg_clf.pickle', 'wb'))
        
        SGD_clf = SklearnClassifier(SGDClassifier())
        SGD_clf.train(self.training_set)
        pickle.dump(SGD_clf, open('./models/SGD_clf.pickle', 'wb'))
        
        SVC_clf = SklearnClassifier(SVC())
        SVC_clf.train(self.training_set)
        pickle.dump(SVC_clf, open('./models/SVC_clf.pickle', 'wb'))
      