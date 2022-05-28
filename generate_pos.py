# -*- coding: utf-8 -*-
"""
@class : GeneratePos

@return
    return part of speech tagging from given list of corpus

@author: Niraj Gautam
"""

import pickle
from helper import text_cleaning
from helper import constant

class GeneratePos:
    
    filePath= constant.constant()
    textClean= text_cleaning.TextClean()
    def __init__(self,quoteFile="quote.txt",nonQuoteFile="nonquote.txt"):
        self.quoteFile=quoteFile
        self.nonQuoteFile=nonQuoteFile
        self.path=self.filePath.path()
    
    
    def generate_pos(self):
        """
        Method generate PART OF SPEECH from corpus and save it in file
        --------------------------------------------------------------
        @return 
        return two list containing qoutePOS and nonQoutePOS
        Ex. return quotePOS , nonQoutePOS
        """
        quoteFile = open (self.path+self.quoteFile, "rb")
        quote = pickle.load(quoteFile)
        
        nonquoteFile = open (self.path+self.nonQuoteFile, "rb")
        nonquote = pickle.load(nonquoteFile)
        
        quote_pos=self.textClean.pos_tag(quote)
        nonquote_pos=self.textClean.pos_tag(nonquote)
        
        with open( self.path+'quote_pos.txt', 'wb') as fh:
           pickle.dump(quote_pos, fh)
        
        with open( self.path+'nonquote_pos.txt', 'wb') as fh:
           pickle.dump(nonquote_pos, fh)
        return quote_pos ,nonquote_pos
