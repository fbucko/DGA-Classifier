"""
Filename: stat_classifier.py
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
This scripts implement statistical classification
for the dga detection. Features are extracted from the
domain and then are comapred to the mean values for 
every known DGA family.
"""

import numpy as np
import pandas as pd
from collections import Counter

#1. Create stat classifier
#2. Functions:
#   -> Read classif matrix
#   -> Extract features
#   -> Compare features
#   -> Classify
class StatClassifier:
    """Class for Statistical classification
    of DGA domains
    """
    def __init__(self, filepath):
        #1. Get classif matrix
        self.__classif_matrix = None #Pandass dataframe
        # self.set_classif_matrix("classif_matrix2")
        self.set_classif_matrix(filepath)
        
    def read_classif_matrix(self, filename:str) -> pd.DataFrame:
        """Method reads the classification matrix
        from csv file (specified by filename) into
        pandas dataframe

        Args:
            filename (str): Name of the csv file where
                            is the classification matrix calculated

        Returns:
            pd.DataFrame: classification matrix in the pandas dataframe
        """
        df = pd.read_csv(filename, index_col=0, header=[0])
        return df
    
    def set_classif_matrix(self, filename:str):
        """Instance method for setting the
        classification matrix

        Args:
            filename (str): Filename of the csv file, where
                            the classif. matrix is calculated
        """
        self.__classif_matrix = self.read_classif_matrix(filename)
        
    def get_classif_matrix(self) -> pd.DataFrame:
        """Instance method which returns
        classification matrix in loaded in
        pandas dataframe

        Returns:
            pd.DataFrame: Classification matrix
        """
        return self.__classif_matrix
    
    def smallest_deviation(self, feature: float, feature_column: pd.Series):
        """The function returns the row with the smallest deviation
        from the specified DGA feature across the entire column. The column
        contains mean values for given feature for all DGA families.

        Args:
            feature (float): DGA feature
            feature_column (pd.Series): Mean values of the DGA feature
                                        for DGA families
        """
        pass
    
    def classify_domain(self, domain_features:np.ndarray[float]) -> str:
        """Instance method classifies the domain according to
        given features

        Args:
            domain_features (np.ndarray[float]): Vector of extracted features

        Returns:
            str: Predicted DGA Family
        """
        # For every feature, find the family with the 
        # lowest deviation
        #1. Compute the deviation for each column, where every column corespond
        #   to certain feature in given domain_features
        classif_matrix = self.get_classif_matrix()
        deviations = np.abs( classif_matrix - domain_features )
        
        #2. Find the row index with lowest deviation for each column
        family_vector = {}
        for column in deviations:
            row_name = deviations[column].idxmin()
            family_vector[column] = row_name
        
        print(family_vector)
        #3. Count the occurence of every family
        print(family_vector.values())
        family_counts = Counter(family_vector.values())
        print(family_counts)
        # Find the row index with the lowest deviation for each column
        # min_deviation_rows = np.argmin(deviations, axis=0)
        # get row names from dataframe for each index
        # row_names = self.__classif_matrix.index.values[min_deviation_rows]
        # print(row_names)
        
        #3. Create dictionary, where key is the feature, and value
        #   is the family name with the lowest deviation
        
        #THE PROBLEM
        #CLassification does not work
        # Find out how much
        # Solution use xgboost -> Combine see the result of the combination
        return None
        
    
if __name__=="__main__":
    stat_classif = StatClassifier()
    print(stat_classif.get_classif_matrix())
    
        
        