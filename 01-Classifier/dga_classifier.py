#!/usr/bin/env python3
"""
Filename: regex_classifier.py
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
This modul implements DGA classifier.
1. Firstly, classifier reads all known regexes for DGA families from
CSV file and tries to classify the domain given as input.
The output of the classification is the family of DGA domain, if the
given domain matches with the database of regexes. 
If the domain does not match any family, it's passed for further 
classification using Machine Learning techniques:
2. Binary classification 
    - First technique is used to detect whether 
      the domain is malicous or benign
    - For this purpose XGBoost method is used
3. Multiclass Classification
    - If the domain is detected as malicious, it's passed
    further for DGA family classification
    - The result of classification can be:
        1. DGA Family
        2. Unknown DGA family -> newly discovered family
        3. Given domain is malicious but is not DGA
    - For Multiclass classification will be tried 2 methods:
        1. Statistical approach
        2. Neural networks:
            a) Perceptron
            b) Transformers
"""
import sys
# sys.path.append("/mnt/d/VUT/Bachelor-thesis" \
#                 "/02-Dataset-preprocessing/" \
#                 "04-Feature-extraction")
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/02-Dataset-preprocessing/04-Feature-extraction")

import numpy as np
import xgboost as xgb
from errors import Err
from known_tld import KnownTLD
from n_grams import N_grams
from argparse import ArgumentParser
from stat_clf import StatClassifier
from regex_clf import RegexClassifier
from binary_dga_clf import BinaryClassifier
from multiclass_dga_clf import MulticlassClassifier
from extract_features import extract_features_2

########################## CONSTANTS ###########################
DGA: int = 1

##################### FUNCTION DEFINITION ######################
def process_input() -> list:
    """
    Function according to script arguments
    reads the input domains intended for DGA classification.
    The domains are read according to specified argument from:
    1. File
    2. Script argument
    3. Stdin
    Function returns the list of the read domains
    
    Returns:
        list: The list of domains intended for classification 
    """
    domains = []
    # 1. Process the input
    parser = ArgumentParser(description="DGA domain classifier")
    parser.add_argument("-f","--file",
                        type=str,
                        help="The the csv filepath, which contains domains for classification")
    parser.add_argument("--domain",
                        action="append",
                        help="The domain intended for classification")
    args = parser.parse_args()
    
    if args.file and args.domain:
        parser.error("Options can't be used in combination")

    # 1.1 If the filename was given, process domain from file
    if args.file:
        with open(args.file, "r") as f:
            domains = [line.strip() for line in f]
    # 1.2 If the domain  
    elif args.domain:
        domains = [ domain.strip() for domain in args.domain ]
    # 1.3 Else process domain from stdin
    else:
        print("Processing from stdin")
        lines = sys.stdin.readlines()
        domains = [line.strip() for line in lines]

    return domains

########################## MAIN ###########################
def main():
    # 1. Process the script input
    domains = process_input()
    if not domains:
        print("No domains were entered", file=sys.stderr)
        sys.exit(Err.EMPTY_INPUT)
    
    # 2. Initialize classifiers
    binary_model_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/binary_classification/"
    regex_model_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/regex_classification/"
    stat_model_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/statistical_classification/"
    xgboost_model_dir ="/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/multiclass_classification/"

    binary_model = "binary_model_concat_labels.pkl"
    regex_model = "regexes_labeled2.txt"
    stat_model = "classif_matrix"
    xgboost_model = "01_multiclass_model_subdomain.pkl"
    
    binary_model_path = binary_model_dir + binary_model
    regex_model_path = regex_model_dir + regex_model
    stat_model_path = stat_model_dir + stat_model
    xgboost_model_path = xgboost_model_dir + xgboost_model
    
    binary_clf = BinaryClassifier(binary_model_path)
    regex_clf = RegexClassifier(regex_model_path)
    stat_clf = StatClassifier(stat_model_path)
    xgboost_clf = MulticlassClassifier(xgboost_model_path)
    
    regex_families = regex_clf.get_regex_families()
    known_subdomain_path = "/mnt/c/Work/Bachelors-thesis/Dataset/Non-DGA/public_suffix_list.dat.txt"
    tlds = KnownTLD(known_subdomain_path)
    known_tlds = tlds.get_tlds()
    
    ngram_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/ngrams/"
    dga_ngram_csv = ngram_dir + "dga-ngram.csv"
    nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
    ngrams = N_grams(dga_ngram_csv=dga_ngram_csv, nondga_ngram_csv=nondga_ngram_csv)
    
    # 3. Classify the given domains
    for domain in domains:
        # 3.1 Extract domain features
        features = np.array(extract_features_2(domain, known_tlds, ngrams)).reshape(1,-1)
        # 3.2 BINARY CLASSIFICATION
        dga_prediction = binary_clf.classify(features)
        
        if dga_prediction == DGA:
            print(f"{domain}: DGA")
            # 3.3 Multiclass classification
            # REGEX
            dga_families_set = regex_clf.classify(domain)
            # STAT
            # dga_family_stat = stat_clf.classify(features)
            # XGBoost
            dga_families_xgboost = xgboost_clf.classify(features)
            
            #ENSEMBLE MODEL (XGBoost + Regex + Stat)
            classification_success = False
            for xgboost_family in dga_families_xgboost:
                # Check if the family is in regex database
                if xgboost_family not in regex_families:
                    #Family is in the DB
                    print(f"{domain} - class: {xgboost_family}")
                    classification_success = True
                    break
                if xgboost_family in dga_families_set:
                    # Classification OK
                    print(f"{domain} - class: {xgboost_family}")
                    classification_success = True
                    break
                # The regex database doesn't contains classifed regex
                # Assumption: Regexes are too genereal
                #             and they should be matched in every scenario
                # Choose the other family that is second most likely and so on...
            
            if not classification_success:
                    print(f"{domain} - class: Unknown")
            # print(dga_families_set)
            # print(dga_family_stat)
            # print(dga_families_xgboost)
            

        else:
            print(f"{domain}: Legit")
            
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    