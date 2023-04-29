"""
Filename: stat_clf.py
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
This scripts serves as an moduled for performance evaluation of the statistical
classification
"""

import sys
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "01-Classifier")
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "02-Dataset-preprocessing/04-Feature-extraction")
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "02-Dataset-preprocessing/05-Multiclass-clf")
import dga_classif_features
import train_models
import stat_clf

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
    
    
    stat_model_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/statistical_classification/"
    stat_model = "classif_matrix"
    stat_model_path = stat_model_dir + stat_model
    clf = stat_clf.StatClassifier(stat_model_path)
    
    y_pred = clf.predict(X_test)
     # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    # roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') # Specify multi-class strategy as 'ovr'
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')

    dga_classif_features.visualize_confusion_matrix_2(confusion)
    # Print the evaluation metrics
    print("Confusion Matrix: ", confusion)
    print("Classification Report: ")
    print(classification)
    # print("ROC AUC Score: ", roc_auc)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

if __name__=="__main__":
    main()