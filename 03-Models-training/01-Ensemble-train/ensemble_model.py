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
The script creates ensemble (cooperative) model from 3 models, which all cooperate in multiclass DGA
classification.
The first model the first model serves to filter out poorly represented
DGA families from strongly represented DGA families in data distribution
Second and Third model serves for the final classification into families.
"""
import sys
sys.path.append("./02-Data-preprocessing/03-Feature-extraction")

import pickle
import n_grams
import known_tld
import train_models
import extract_features
import dga_classif_features
import numpy as np
from dataset_paths import DatasetPaths

class EnsembleClassifier:
    def __init__(self,
                 binary_model_path:str=None,
                 model2_path:str=None,
                 model3_path:str=None,
                 label2_path:str=None,
                 label3_path:str=None):
        model_dir = "./04-Models/ensemble_classification/01-Models/" 
        label_dir = "./04-Models/ensemble_classification/02-Labels/"
    
        if not binary_model_path:
            binary_model_path = model_dir + "01-binary-clf.pkl"

        if not model2_path:
            model2_path = model_dir + "02-below-median-clf.pkl"

        if not model3_path:
            model3_path = model_dir + "03-above-median-clf.pkl"
        
        if not label2_path:
            label2_path = label_dir + "02-below-median-labels.json"
        
        if not label3_path:
            label3_path = label_dir + "03-above-median-labels.json"

        # Load the binary classifier from pickle file
        with open(binary_model_path, 'rb') as f:
            self.binary_model = pickle.load(f)
        self.labels = dga_classif_features.get_labels(
            DatasetPaths().get_stat_clf_labels_dir()
        )
        # Load the multiclass Model 2 from pickle file
        with open(model2_path, 'rb') as f:
            self.model2 = pickle.load(f)
        self.label2_labels = train_models.load_labels(label2_path)
        
        # Load the multiclass Model 3 from pickle file
        with open(model3_path, 'rb') as f:
            self.model3 = pickle.load(f)
        self.label3_labels = train_models.load_labels(label3_path)
    
    def get_labels(self, label_id:bool=False) -> dict:
        if not label_id:
            return self.label2_labels
        else:
            return self.label3_labels
    
    def convert_to_labels(self, family_prediction:list[str]) -> list[int]:
        """Converts list of dga family names into labels
        (int from 0 to 92).

        Args:
            family_prediction (list[str]): List of DGA predicted families

        Returns:
            list[int]: List of DGA predicted labels
        """
        return [self.labels[family] for family in family_prediction]

    def predict(self, X):
        # Predict binary outcome using the binary classifier
        binary_predictions = self.binary_model.predict(X)
        # Initialize an array to store final predictions
        final_predictions = []
        
        # Loop through each binary prediction
        for i in range(len(binary_predictions)):
            if binary_predictions[i] == 0:
                # If binary prediction is 0, use Model 2 for multiclass prediction
                row = X.iloc[i]
                prediction = self.model2.predict(X.iloc[i,:].values.reshape(1, -1))
                keys = [key for key, val in self.label2_labels.items() if val == prediction]
                prediction = keys[0]
            else:
                row = X.iloc[i,:]
                # If binary prediction is 1, use Model 3 for multiclass prediction
                prediction = self.model3.predict(X.iloc[i,:].values.reshape(1, -1))
                # Find keys associated with the given value
                keys = [key for key, val in self.label3_labels.items() if val == prediction]
                prediction = keys[0]
                
            final_predictions.append(prediction)
        
        return self.convert_to_labels(final_predictions)

def main():
    model_dir = "./04-Models/ensemble_classification/01-Models/"
    label_dir = "./04-Models/ensemble_classification/02-Labels/"
    binary_model_path = model_dir + "01-binary-clf.pkl"
    model2_path = model_dir + "02-below-median-clf.pkl"
    model3_path = model_dir + "03-above-median-clf.pkl"
    label2_path = label_dir + "02-below-median-labels.json"
    label3_path = label_dir + "03-above-median-labels.json"
    
    clf = EnsembleClassifier(binary_model_path,
                             model2_path,
                             model3_path,
                             label2_path,
                             label3_path)
    
    domain = "wdgmgvpvztygk5q.com" #bedep
    domain = "wevydrkvywxqfsul.ru" # Blackhole
    domain = "ab6d54340c1a.com" #ccleaner
    
    known_subdomain_path = DatasetPaths().get_known_tlds_path()
    tlds = known_tld.KnownTLD(known_subdomain_path)
    known_tlds = tlds.get_tlds()
    
    ngram_dir = DatasetPaths().get_ngrams_dir()
    dga_ngram_csv = ngram_dir + "dga-ngram.csv"
    nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
    ngrams = n_grams.N_grams(dga_ngram_csv=dga_ngram_csv, nondga_ngram_csv=nondga_ngram_csv)
    
if __name__=="__main__":
    main()