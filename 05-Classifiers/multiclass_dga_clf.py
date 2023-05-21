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

sys.path.append("./03-Models-training")
from model_paths import ModelPaths

class MulticlassClassifier:
    """
    Class for dividing DGA domains
    into malware families. Class implements methods
    to work with classification model. Such as 
    reading the model, classifying domains etc.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the multiclass classifier -> read
        the created model from directory
        Args:
            model_path (str): Path to the binary model
        """
        self.__model = None
        self.__labels = {}
        self.read_model(model_path)
        self.read_labels()
        
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
            sys.exit(Err.EMPTY_MULTICLASS_CLF)
    
    def get_labels(self) -> dict:
        """
        The method returns the names of the families
        with the associated labels that are used when
        training the model

        Returns:
            dict: Dictionary of the family names with associated labels
        """
        return self.__labels
    
    def read_labels(self) -> dict:
        """
        Read labels which were generated while
        creating model for decoding model output

        Args:
            labels_file (str): file where labels are stored
        """
        labels_dir = ModelPaths().get_xgb_multiclass_clf_labels_path()
        labels_path = labels_dir + "dga_model_labels.pkl"  
        with open(labels_path, "rb") as f:
            # read the dictionary of the labels
            self.__labels = pickle.load(f)
        
        try:
            assert self.__labels != None, "Unable to read labels from specified path:\n" \
                                        f"{labels_path}"
        except AssertionError as err:
            print(err, file=sys.stderr)
            sys.exit(Err.EMPTY_MULTICLASS_LABELS)
             
    def labels_to_families(self, labels: list[int]) -> list:
        """
        Method maps a list of integers(labels) that correspond
        to individual dga families and returns the list of mapped families.

        Args:
            labels (list[int]): List of labels

        Returns:
            list: List of family names
        """
        labels_dict = self.get_labels()
        families = [key 
                    for label in labels 
                        for key, value in labels_dict.items()
                            if label == value
        ]
        return families
    
    def classify(self, domain_features:np.ndarray[float]) -> dict:
        """
        Method for classyfing the DGA domain into malware families,
        according to given domain features 

        Args:
            domain_features (np.ndarray[float]): Features extracted from domain
                                                 (domain lenght, consonant ratio, ...)

        Returns:
            dict: Dictionary where keys are families and values are probabilities
        """
        # 1. Get predicted probabilities for each label
        probabilities = self.__model.predict_proba(domain_features)
        # 2. Get sorted indices of labels according to probabilities
        sorted_indices = np.argsort(-probabilities, axis=1)
        # 3. Get labels sorted according to probabilities
        sorted_labels = self.__model.classes_[sorted_indices]
        # 4. Reduce the array dimensionality
        sorted_labels = np.squeeze(sorted_labels)
        # 5. Convert labels to family names
        return self.labels_to_families(sorted_labels)
        