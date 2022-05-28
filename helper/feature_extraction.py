# -*- coding: utf-8 -*-
"""
@class : FeatureExtraction 

@method : find_features

@parameter 
    document string

@author: Niraj Gautam
"""

from nltk.tokenize import word_tokenize
from helper import constant
import pickle
class Featurextraction:
    
    filePath= constant.constant()
    word_feature_file="word_features.txt"
    
    def __init__(self):
        self.word_features = open (self.filePath.path() + self.word_feature_file, "rb")
        self.word_features = pickle.load(self.word_features)
        
        
        
        
    def find_feature(self,document):
        """
        @parameter
            document: string 
        @return 
            tokenized features list
        """
        words = word_tokenize(document)
        features = {}
        for w in self.word_features:
            features[w] = (w in words)
        return features
        
