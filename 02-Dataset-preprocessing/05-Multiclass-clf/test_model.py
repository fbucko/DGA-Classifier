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
Script used for testing trained models.
Input: Dataset for testing
Output: ML Metrics like precision, accuracy, k-fold-cross validation and so on...
"""

import sys
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "02-Dataset-preprocessing/04-Feature-extraction")
import train_models
import dga_classif_features
import ensemble_model
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

def main():
    dir_path= "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset"\
              "/dgarchive_datasets/02-Proportion-pick/"
    df = dga_classif_features.read_and_label_dataset(dir_path, label_column="dga_labels")
    df = df.head(n=100000)
    feature_columns = ["domain_len",
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
        "www_flag"] 
    dataset_with_features_df = train_models.compute_features(df, feature_columns)
    print(dataset_with_features_df)
    
    X_test = dataset_with_features_df[feature_columns]
    y_test = dataset_with_features_df["dga_labels"]
    
    print(X_test)
    print(y_test)
    
    model_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/02-Dataset-preprocessing/05-Multiclass-clf/01-Models/"
    label_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/02-Dataset-preprocessing/05-Multiclass-clf/02-Labels/"
    binary_model_path = model_dir + "01-binary-clf.pkl"
    model2_path = model_dir + "02-below-median-clf.pkl"
    model3_path = model_dir + "03-above-median-clf.pkl"
    label2_path = label_dir + "02-below-median-labels.json"
    label3_path = label_dir + "03-above-median-labels.json"
    clf = ensemble_model.EnsembleClassifier(
        binary_model_path,
        model2_path,
        model3_path,
        label2_path,
        label3_path)
    
    y_pred = clf.predict(X_test)
     # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    # roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') # Specify multi-class strategy as 'ovr'
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')

    # Print the evaluation metrics
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: ", confusion)
    print("Classification Report: ")
    print(classification)
    print("F1 Score: ", f1)
    # print("ROC AUC Score: ", roc_auc)
    print("Precision: ", precision)
    print("Recall: ", recall)

if __name__=="__main__":
    main()