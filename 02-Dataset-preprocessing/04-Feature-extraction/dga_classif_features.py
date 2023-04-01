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
import pandas as pd
import xgboost as xgb
from extract_features import extract_features
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        file[len(prefix):-len(suffix)]
        for file in files
        if file.startswith(prefix)
    ]
    # 2. Create the dicitionay with family names
    #    and assign labels to them
    families = {}
    for idx, dga in enumerate(dga_families):
        families[dga] = idx
    
    return families

if __name__=="__main__":
    # 1. Get the labels for every family to create the dataset
    dir_path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_datasets/02-Proportion-pick/"
    family_labels = get_labels(dir_path)
    
    # 2. Read the files into dataframe and assign 
    #    them the label, corresponding to the family
    dfs = []
    prefix = "01-"
    suffix = ".csv"
    file_list = glob.glob(dir_path + prefix + "*" + suffix)
    for file in file_list:
        # 3. Assign the labels
        filename = os.path.basename(file)
        family = filename[len(prefix):-len(suffix)]
        label = family_labels[family]
        df = pd.read_csv(file, names=["domain"])
        df["dga_family"] = label                
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # 4. Compute the features- use parallel processing for each chunk
    feature_values = df['domain'].apply(extract_features)
    
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
        "first_digit_flag",
        "norm_entropy"]
    feature_df = pd.DataFrame(feature_values.tolist(), columns=feature_names)

    # concatenate the feature dataframe with the original dataframe
    final_df = pd.concat([df, feature_df], axis=1)
    print(final_df)
    
    # 5. Train the model    
    # CREATE XGBOOST MODEL
    model_name = "multiclass_model.pkl"
    # Split the data
    X = final_df[["domain_len",
        "tld_len",
        "sld_len",
        "max_consonant_len",
        "sld_digits_len",
        "unique_chars",
        "digit_ratio",
        "consonant_ratio",
        "non_alfa_ratio",
        "hex_ratio",
        "first_digit_flag",
        "norm_entropy"]]
    y = final_df["dga_family"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    if not os.path.isfile(model_name):
        print("The model does not exist, creating one")
        
    

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # le = LabelEncoder()
        # y_train = le.fit_transform(y_train)
        
        # Train the XGBoost model
        params = {'objective': 'multi:softmax', 'num_class': 93, 'max_depth': 3, 'learning_rate': 0.2, 'n_estimators': 100}
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
    
    # 6. Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    # roc_auc = roc_auc_score(y_test, y_pred, multi_class="over", average="macro")
    
    # scores = cross_val_score(model, X, y, cv=5)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    # print('ROC AUC:', roc_auc)
    # print('Scores:', scores.mean())

    #Export model to file
    # with open('model.pkl', 'wb') as f:
        # pickle.dump(model, f)