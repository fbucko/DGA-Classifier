#!/usr/bin/env python3
"""
Filename: regex_clf.py
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
This modul implements DGA classifier based on regex classification.
Classifier reads all known regexes for DGA families from CSV file and
tries to classify the domain given as input.
The output of the classification is the family of DGA domain, if the
given domain matches with the database of regexes. 
If the domain does not match any family, it's passed for further 
classification using Machine Learning techniques.
"""

import re
import pandas as pd
class RegexClassifier:
    """
    Class for classifying DGA domains using
    regular expressions.
    """
    def __init__(self, filename) -> None:
        self.__dga_regex_df = None # Pandas dataframe
        self.__dga_regex_families = set()
        self.read_regexes(filename)
        self.read_regex_families()
        
    def read_regexes(self, filename:str) -> pd.DataFrame:
        """
        Instance method for reading regexes from given file

        Args:
            filename (str): Filepath

        Returns:
            pd.Dataframe: If the work with file was successful,
            returns pandas dataframe.
        """
        # 1. Read DGA regex database from CSV file
        columns = ["DGA Family", "Regex"] # Be careful on the header
        df = pd.read_csv(filename, names=columns)
        self.__dga_regex_df = df
    
    def read_regex_families(self) -> set:
        """
        Creates the set of all family names
        that have regex from pandas dataframe column

        Returns:
            set: Set of all families that have regex
        """
        if self.__dga_regex_df is not None:
            self.__dga_regex_families = set(self.__dga_regex_df["DGA Family"])
            
    def get_regex_families(self) -> set:
        """
        Returns the set of all family names
        that have regex
        
        Returns:
            set: Set of all families that have regex
        """
        return self.__dga_regex_families
    
    def classify(self, domain:str) -> list:
        """
        Instance method for classifying domain with 
        regex matching.
        If the regex classification is successful, 
        method returns DGA Family which matches the
        given domain.
        If the domain does not match any regex from
        regex database it returns None

        Args:
            domain (str): Domain given for classification
        
        Returns:
            list: Method returns the list of matched DGA families.
                  If no families were matched, empty list is returned
        """
        matched_dga_set = set()
        for idx, regex in enumerate(self.__dga_regex_df["Regex"]):
            match = re.match(regex, domain)
            if match:
                matched_dga_set.add(self.__dga_regex_df['DGA Family'].iloc[idx])
        
        return matched_dga_set
   
if __name__ == "__main__":
    print(" Regex classif Run from terminal")


