#!/usr/bin/env python3
"""
Filename: binary_clf.py
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
This modul implements DGA binary classifier.
Binary classification 
    - First technique is used to detect whether 
      the domain is malicous or benign
    - For this purpose XGBoost method is used
"""
import sys
import pickle
import numpy as np
from errors import Err

class BinaryClassifier:
    """
    Class for filtering benign domains
    from DGA domains. Class implements methods
    to work with classification model. Such as 
    reading the model, classifying domains etc.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the binary classifier -> read
        the created model from directory
        Args:
            model_path (str): Path to the binary model
        """
        self.__model = None
        self.read_model(model_path)
        
    def read_model(self, model_path: str):
        """
        Read the model for binary classification
        
        Args:
            model_path (str): Path to the binary model
        """
        with open(model_path, 'rb') as f:
            self.__model = pickle.load(f)
            
        try:
            assert self.__model != None, "Unable to read model from specified path:\n" \
                                        f"{model_path}"
        except AssertionError as err:
            print(err, file=sys.stderr)
            sys.exit(Err.EMPTY_BINARY_CLF)
    
    def classify(self, domain_features:np.ndarray[float]) -> int:
        """
        Method for classyfing the domain, whether it's DGA domain
        or non-DGA domain, according to given domain features 

        Args:
            domain_features (np.ndarray[float]): Features extracted from domain
                                                 (domain lenght, consonant ratio, ...)

        Returns:
            int: 1 If the given features indicate that it is a DGA domain
                 0 If the given features indicate that it is a non-DGA domain
        """
        return self.__model.predict(domain_features)