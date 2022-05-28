# -*- coding: utf-8 -*-
"""
EnsembleClassider class takes classifier as parameter 
and ensemble them together 

@return 
    classification based on majority of votes
@author: Niraj Gautam
"""

from nltk.classify import ClassifierI
from statistics import mode

# Defininig the ensemble model class 
class EnsembleClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    # a simple measurement the degree of confidence in the classification 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf