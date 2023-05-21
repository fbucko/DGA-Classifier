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
Module implements object which will hold all the features for training and
testing binary and multiclass models.
"""

class Features:
    def __init__(self) -> None:
        self.__domain_column = "domain"
        self.__feature_columns = [
            "domain_len",
            "tld_len",
            "sld_len",
            "max_consonant_len",
            "sld_digits_len",
            "unique_chars",
            "digit_ratio",
            "consonant_ratio",
            "non_alfa_ratio",
            "hex_ratio",
            "dictionary_match",
            "dga_ngram_ratio",
            "nondga_ngram_ratio",
            "first_digit_flag",
            "well_known_tld",
            "norm_entropy",
            "subdomains_count",
            "www_flag",
        ] 
        self.__binary_label_column = "is_dga"
        self.__multiclass_label_column = "dga_family"
    
    # Setter methods 
    def set_domain_column(self, new_domain_column_name: str) -> None:
        self.__domain_column = new_domain_column_name
    
    def set_feature_columns(self, new_feature_column: list) -> None:
        self.__feature_columns = new_feature_column
        
    def set_binary_label_column(self, new_label:str) -> None:
        self.__binary_label_column = new_label
    
    def set_multiclass_label_column(self, new_label:str) -> None:
        self.__multiclass_label_column = new_label
    
    # Getter methods
    def get_domain_column(self) -> str:
        return self.__domain_column
    
    def get_feature_columns(self) -> list:
        return self.__feature_columns
    
    def get_binary_label_column(self):
        return self.__binary_label_column
    
    def get_multiclass_label_column(self):
        return self.__multiclass_label_column
    
        
    