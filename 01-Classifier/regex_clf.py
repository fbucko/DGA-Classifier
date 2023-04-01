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
        self.read_regexes(filename)
        
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
        columns = ["DGA Family", "Regex"]
        df = pd.read_csv(filename, names=columns)
        self.__dga_regex_df = df
        
    def classify(self, domain:str) -> str:
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
            int: If the domain was matched with one of the regexes,
                 method returns DGA family.
                 Else returns None.
        """
        # 1. Iterate through list of regexes
        for idx, regex in enumerate(self.__dga_regex_df["Regex"]):
            match = re.match(regex, domain)
            if match:
                print("OK: Pattern found")
                print(f"Family: {self.__dga_regex_df['DGA Family'].iloc[idx]}")
            else:
                pass
                #print("BAD: Pattern not found ")
            # 2. Try to match regex with given domain
            # Decide what are the possible scenarios:
            # a) Use re.match() function
            # b) Use re.search() function
            # c) Use re.findall() function
        # 3. Check the result of matching
        
        # PROBLEMS ->
        #   1. One domain matches multiple regexes, which one is correct?
        #       -> The classification can't be complited the problem
        #          will be passed to ML method, with the list of families.
        #          After that we will add values (from regex classification)
        #          to probability scores from ML classification
        #   2. Regex matches also valid domains
        
if __name__ == "__main__":
    print(" Regex classif Run from terminal")
    # filepath = "/mnt/d/VUT/Bachelor-thesis/Dataset/regexes.txt"
    filepath = "/mnt/d/VUT/Bachelor-thesis/02-Dataset-preprocessing/01-Regex-classif/regexes.txt"
    
    classifier = RegexClassifier(filepath)
    # Bamital
    # classifier.classify("cd8f66549913a78c5a8004c82bcf6b01.info") 
    
#1. Proces input
#2. Read regex database from csv
#3. Classify the given domain


