# -*- coding: utf-8 -*-
"""
@class : TestModel

@return
    prediction of testing set
@author: Niraj Gautam
"""
import model
import ensemble

class TestModel:
    
    def __init__(self,testing_set):
        self.testing_set=testing_set
        
    
    def test(self):
        """
        @return 
            prediction list of dictionary
        """
        m=model.Model()
        ONB_Clf = m.load_model('./models/ONB_clf.pickle')
        MNB_Clf = m.load_model('./models/MNB_clf.pickle')
        BNB_Clf = m.load_model('./models/BNB_clf.pickle')
        LogReg_Clf = m.load_model('./models/LogReg_clf.pickle')
        SGD_Clf = m.load_model('./models/SGD_clf.pickle')
        SVC_clf = m.load_model('./models/SVC_clf.pickle')
        
        ensemble_clf = ensemble.EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf,SVC_clf)
        
        feature_list = [f[0] for f in self.testing_set]
        ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]
        
        return ensemble_preds