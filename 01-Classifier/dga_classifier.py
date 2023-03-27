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
sys.path.append("/mnt/d/VUT/Bachelor-thesis/02-Dataset-preprocessing/04-Feature-extraction")
import pickle
import numpy as np
import xgboost as xgb
from extract_features import extract_features 
from argparse import ArgumentParser
from regex_classifier import RegexClassifier
from stat_classifier import StatClassifier

if __name__ == "__main__":
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
        domains =[ domain.strip() for domain in args.domain]
    # 1.3 Else process domain from stdin
    else:
        print("Processing from stdin")
        lines = sys.stdin.readlines()
        domains = [line.strip() for line in lines]
      
    model_path = "/mnt/d/VUT/Bachelor-thesis/02-Dataset-preprocessing/04-Feature-extraction/model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)  

    # 2. Call (ML) binary classifier for separation DGA and non DGA domains
    # Init classifiers
    # Regex
    filepath = "/mnt/d/VUT/Bachelor-thesis/02-Dataset-preprocessing/01-Regex-classif/regexes.txt"
    regex_clf = RegexClassifier(filepath)
    stat_clf = StatClassifier("classif_matrix2")
    
    for domain in domains:
        features = np.array(extract_features(domain)).reshape(1,-1)
        print(features)
        print(features.dtype)
        
        prediction = model.predict(features)
        print(prediction)
        if prediction == 1:
            print(f"{domain}: DGA")
            # 3. Call methods for classification
            DGA_families = regex_clf.classify(domain)
            DGA_family = stat_clf.classify_domain(features)
            print(DGA_families)
            print(DGA_family)
        else:
            print(f"{domain}: Legit")
            
        
    
  
    
    
    # 2. Call methods from regex classification module
    # 3. Call machine learning models
    # 3.1 
    # 4. Return formatted result
    
    
    
    
    
    
    
    