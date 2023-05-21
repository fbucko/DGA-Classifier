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
Purpose of this script is to read the database of DGA and Non-DGA ngrams
from csv file and check whether .
"""
import pandas as pd
import extract_ngrams
from dataset_paths import DatasetPaths
class N_grams:
    def __init__(self, dga_ngram_csv:str=None, nondga_ngram_csv:str=None):
        """Instance init method

        Args:
            dga_ngram_csv (str): filepath to the dga-ngram csv file
            non_dga_ngram_csv (str): filepath to the non-dga-ngram csv file
        """
        self.__dga_ngrams = set()
        self.__nondga_ngrams = set()
        ngram_dir = DatasetPaths().get_ngrams_dir()

        if dga_ngram_csv:
            self.load_dga_ngrams(dga_ngram_csv)
        else:
            dga_ngram_csv = ngram_dir + "dga-ngram.csv"
            self.load_dga_ngrams(dga_ngram_csv)
            
        if nondga_ngram_csv:
            self.load_nondga_ngrams(nondga_ngram_csv)
        else:
            nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
            self.load_nondga_ngrams(nondga_ngram_csv)
    
    def load_dga_ngrams(self, dga_ngram_csv:str):
        """Method loads the ngrams from csv file and returns
        the set of DGA ngrams. This function is used for
        initializing the instance attirbute

        Args:
            dga_ngram_csv (str): Filepath to the csv file
        """
        ngrams_df = pd.read_csv(dga_ngram_csv)
        self.__dga_ngrams = set(ngrams_df["ngram"])


    def load_nondga_ngrams(self, nondga_ngram_csv:str):
        """Method loads the ngrams from csv file and returns
        the set of benign ngrams. This function is used for
        initializing the instance attirbute

        Args:
            nondga_ngram_csv (str): Filepath to the csv file
        """
        ngrams_df = pd.read_csv(nondga_ngram_csv)
        self.__nondga_ngrams = set(ngrams_df["ngram"])
        
    def count_dga_ngram(self, ngrams:set) -> int:
        """
        Method counts how many ngrams belong to 
        DGA ngram set.

        Args:
            ngram (set): set of extracted n-grams

        Returns:
            int: The number of elements in the intersection
        """
        intersection = self.__dga_ngrams.intersection(ngrams)
        return len(intersection)
        
    def count_nondga_ngram(self, ngrams:str) -> int:
        """
        Method counts how many ngrams belong to 
        benign ngram set.

        Args:
            ngram (set): set of extracted n-grams

        Returns:
            int: The number of elements in the intersection
        """
        intersection = self.__nondga_ngrams.intersection(ngrams)
        return len(intersection)
    
    def dga_ngram_ratio(self, domain: str, n: int) -> float:
        """
        Methods calculates the dga ngram ratio.

        Args:
            domain (str): Domain name
            n (int): n-gram specifier (2,3,4-gram etc.)

        Returns:
            float: ngram ratio
        """
        domain_ngrams = extract_ngrams.extract_ngrams(domain, n)
        dga_ngram_count = self.count_dga_ngram(domain_ngrams)
        return dga_ngram_count / len(domain_ngrams) if len(domain_ngrams) > 0 else 0.0
            
    def nondga_ngram_ratio(self, domain: str, n:int) -> float:
        """
        Methods calculates the benign ngram ratio.

        Args:
            domain (str): Domain name
            n (int): n-gram specifier (2,3,4-gram etc.)

        Returns:
            float: ngram ratio
        """
        domain_ngrams = extract_ngrams.extract_ngrams(domain, n)
        nondga_ngram_count = self.count_nondga_ngram(domain_ngrams)
        return nondga_ngram_count / len(domain_ngrams) if len(domain_ngrams) > 0 else 0.0
    
    def dga_ngram_avg_ratio(self, domain:str) -> float:
        """
        Methods calculates the dga ngram ratio.

        Args:
            domain (str): Domain name

        Returns:
            float: ngram ratio
        """
        ngram_values = []
        for n in 2,3,4:
            ngram_ratio = self.dga_ngram_ratio(domain, n)
            if ngram_ratio:
                ngram_values.append(ngram_ratio)
            
        return sum(ngram_values) / len(ngram_values) if len(ngram_values) > 0 else 0.0

    def nondga_ngram_avg_ratio(self, domain: str) -> float:
        """
        Methods calculates the benign ngram ratio.

        Args:
            domain (str): Domain name

        Returns:
            float: ngram ratio
        """
        ngram_values = []
        for n in 2,3,4:
            ngram_ratio = self.nondga_ngram_ratio(domain, n)
            if ngram_ratio:
                ngram_values.append(ngram_ratio)
        
        return sum(ngram_values) / len(ngram_values) if len(ngram_values) > 0 else 0.0