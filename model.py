# -*- coding: utf-8 -*-
"""
@class : Model

@method : load_model

@return : classifier's object

@author: Niraj Gautam
"""

import pickle

class Model:
    
    def load_model(self,file_path):
        """
            @parameter
                file_path : file path of  pickled classifier to load
            @return : classifier's object    
        """
        classifier_f = open(file_path, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        return classifier