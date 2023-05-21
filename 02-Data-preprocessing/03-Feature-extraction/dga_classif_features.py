#!/usr/bin/env python3
"""                                                                                                                                                                 
Filename: dga_classif_features.py
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
Purpose of this script is to compute specified features
for DGA domains in datasets and label them according to family they come from.
Features will be used for DGA binary classification.
Selected features:
1. Total domain length
2. TLD length
3. SLD length
4. The length of the longest consonant sequence in SLD
5. Number of digits in SLD
6. Number of unique characters in the domain string (TLD + SLD)
7. Digit ratio -> number of digits divided by string length
8. Consonant ratio -> number of consonants divided by string length
9. Non-alfanumeric ratio
10. Hexadecimal ratio
11. Flag of beginning number
12. Flag of malicious domain -> : “study”, “party”, “click”, “top”, “gdn”, “gq”, “asia”,
“cricket”, “biz”, “cf”.
13. Normalized entropy of the string
14. N - gram features
    -> extract ngrams from datasets -> 2,3,4 gram -> build dictionaries -> save them to separate file
    -> Create 2 files for bening N-grams and for DGA N-grams
    -> Calculate the ratio for 2,3,4 gram by counting the number of occurence in Benign and DGA hashtables
       and dividing the amount of n-grams of the string
    -> calculate the average value for 2,3,4 ngrams for Benign and DGA classes
15. DGA similarity -> abs(DGA n-gram average - Non-dga average)
16. Dictionary matching ratio -> length of matched words / total domain length
17. Contains www
18. Subdomain count 
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from n_grams import N_grams
from known_tld import KnownTLD
from dataset_paths import DatasetPaths

from extract_features import extract_features_from_concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def get_labels(dir_path:str) -> dict:
    """The function lists all the files in the given
    path parameter. The directory should contain
    files, where every file represents DGA Family
    and contains data of the specific DGA family examples.
    The names of the files are also name of the DGA Families.
    Function extracts the DGA family names from the file names
    and return them in the dictionary with assigned label, which
    will be used later for labeling the data.
    All csv files that contain a specific instance of the family
    have the prefix "01-".

    Args:
        dir_path (str): The directory, where the files represent
        DGA family

    Returns:
        dict: The dictionary with the family names and the labels 
    """
    # 1. List all the files in the given directory
    prefix = "01-"
    suffix = ".csv"
    files = os.listdir(dir_path)
    dga_families = [ 
        file[len(prefix):-len(suffix)] # Remove prefix and suffix from filename
        for file in files
        if file.startswith(prefix)
    ]
    dga_families.sort()
    # 2. Create the dicitionay with family names
    #    and assign labels to them
    families = {}
    for idx, dga in enumerate(dga_families):
        families[dga] = idx
    
    return families

def save_labels(labels: dict, filename: str):
    """
    Function saves generated labels for further
    processing as part of the DGA classification
    into pickle file.

    Args:
        labels (dict): labels that will be saved
        filename (str): name of the pickle file
    """
    with open(filename, "wb") as f:
        pickle.dump(labels, f)

def read_and_label_dataset(dataset_dir:str, label_column:str):
    """Function reads the files in specified directory.
    Each file contains samples for a specific DGA family.
    Since we want to combine all families into one dataframe,
    we have to label them.

    Args:
        dataset_dir (str): _description_
    """
    dfs = []
    prefix = "01-"
    suffix = ".csv"
    family_labels = get_labels(dataset_dir)
    file_list = glob.glob(dataset_dir + prefix + "*" + suffix)
    for file in file_list:
        # 3. Assign the labels
        filename = os.path.basename(file)
        family = filename[len(prefix):-len(suffix)] # Remove prefix and suffix from filename
        label = family_labels[family] 
        df = pd.read_csv(file, names=["domain"])
        df[label_column] = label                
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) #concatenated dataframe

def visualize_confusion_matrix(confusion_matrix:np.ndarray):
    # 1. Create heatmap for confusion matrix
    # plt.figure(figsize=(30, 30))  # specify a larger figure size
    # sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", cbar=False)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.show()
    # Define a diagonal-only plot using Plotly
    ticks=np.linspace(0, 92,num=93)
    plt.figure(figsize=(20,15))
    plt.imshow(confusion_matrix, interpolation='none')
    plt.colorbar()
    plt.xticks(ticks,fontsize=6)
    plt.yticks(ticks,fontsize=6)
    plt.grid(True)
    plt.show()

def k_fold_cros_val_multiclass(model, X, y, num_folds):
    # define the k-fold cross-validation object
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # perform k-fold cross-validation and return the accuracy scores
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    # print the mean and standard deviation of the accuracy scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

def visualize_confusion_matrix_2(confusion_matrix:np.ndarray):
    total_per_class = np.sum(confusion_matrix, axis=1)

    # calculate the number of misclassified instances per class
    misclassified_per_class = total_per_class - np.diag(confusion_matrix)

    # calculate the misclassification rate per class
    misclassification_rate_per_class = misclassified_per_class / total_per_class

    # print the misclassification rate per class
    number_of_classes = confusion_matrix.shape[0]
    print(number_of_classes)
    for i in range(number_of_classes):
        print(f"Class {i}: {misclassification_rate_per_class[i]:.2%}")
    
    ticks=np.linspace(0, number_of_classes - 1,num=number_of_classes)
    cm_normalized = np.round(confusion_matrix/np.sum(confusion_matrix, axis=1).reshape(-1,1),2)
    sns.heatmap(cm_normalized,
                cmap="Greens",
                annot=False,
                xticklabels=ticks,
                yticklabels=ticks)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
def main():
    # 1. Get the labels for every family to create the dataset
    dir_path = DatasetPaths().get_proportion_picked_dataset(2)
    family_labels = get_labels(dir_path)
    labels_path = "dga_model_labels.pkl"
    save_labels(family_labels, labels_path)
    
    # 2. Read the files into dataframe and assign 
    #    them the label, corresponding to the family
    df = read_and_label_dataset(dir_path, label_column="dga_family")
    
    # 4. Compute the features - use parallel processing for each chunk
    known_subdomain_path = DatasetPaths().get_known_tlds_path()
    tlds = KnownTLD(known_subdomain_path)
    known_tlds = tlds.get_tlds()
    
     
    ngram_dir = DatasetPaths().get_ngrams_dir()
    dga_ngram_csv = ngram_dir + "dga-ngram.csv"
    nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
    ngrams = N_grams(dga_ngram_csv=dga_ngram_csv, nondga_ngram_csv=nondga_ngram_csv)
    feature_values = df['domain'].apply(lambda x:extract_features_from_concat(x, known_tlds, ngrams))
    
    # convert the list of feature values into a pandas dataframe
    feature_names = ["domain_len",
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
    feature_df = pd.DataFrame(feature_values.tolist(), columns=feature_names)

    # concatenate the feature dataframe with the original dataframe
    final_df = pd.concat([df, feature_df], axis=1)
    # 5. Train the model    
    # CREATE XGBOOST MODEL
    model_name = "01_multiclass_model_subdomain.pkl"
    # Split the data
    X = final_df[feature_names]
    y = final_df["dga_family"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, shuffle=True)
    
    train_unique = sorted(y_train.unique())
    test_unique = sorted(y_test.unique())
    
    if not os.path.isfile(model_name):
        print("The model does not exist, creating one")
        # Train the XGBoost model
        params = {'objective': 'multi:softmax', 'num_class': 93, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100}
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Check if the model exists
        print("Saving the model")
        with open(model_name, 'wb') as f:
            pickle.dump(model, f)
    else:
        # LOAD THE MODEL
        # Load the saved XGBoost model
        print("Model exists")
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
    
    xgb.plot_importance(model)
    

    k_fold_cros_val_multiclass(model, X, y, num_folds=5)
    
    # 6. Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
    cm = confusion_matrix(y_test, y_pred)
    # roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') # Specify multi-class strategy as 'ovr'
    
    # get gain score
    # clf je XGBClassifier
    score = model.get_booster().get_score(importance_type='gain')
    sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    print(sorted_score)    
    
    # scores = cross_val_score(model, X, y, cv=5)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    visualize_confusion_matrix_2(cm)

    #Export model to file
    # with open('model.pkl', 'wb') as f:
        # pickle.dump(model, f)

if __name__=="__main__":
    main()
    # visualize_confusion_matrix("nice")