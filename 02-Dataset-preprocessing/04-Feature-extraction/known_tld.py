#!/usr/bin/env python3
"""                                                                                                                                                                  3 Filename: regex_classifier.py
Author: Filip Bučko
Date: March 3, 2023
License: MIT License

Copyright (c) 2023 Filip Bučko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Description:
Purpose of this script is to read the database of the known tlds
from txt file and return them in set.
"""
import math
import pickle
import tldextract
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score


class KnownTLD:
    def __init__(self, filename:str) -> None:
        self.__tlds = set()
        self.read_known_tld(filename)
        
    def read_known_tld(self, filename:str):
        """
        Method read all known tlds from file into instance
        set variable

        Args:
            filename (str): The name of the known tld database
        """
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue  # skip empty and comment lines
                self.__tlds.add(line)  # add the valid TLD to the set

        # print(self.__tlds)  # print the set of TLDs

    def get_tlds(self):
        """Returns the instance variable
        of known tlds
        """
        return self.__tlds
    
if __name__=="__main__":
    #main()
    domain = "www.google.edu.gd"