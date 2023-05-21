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
for domains in datasets. Features will be used for DGA
binary classification.
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
12. Flag of well known domain 
13. Normalized entropy of the string 
14. N - gram features
    -> extract ngrams from datasets -> 2,3,4 gram -> build dictionaries -> save them to separate file
    -> Create 2 files for bening N-grams and for DGA N-grams
    -> Calculate the ratio for 2,3,4 gram by counting the number of occurence in Benign and DGA hashtables
       and dividing the amount of n-grams of the string
    -> calculate the average value for 2,3,4 ngrams for Benign and DGA classes
15. DGA similarity -> abs(DGA n-gram average - Non-dga average)
16. Dictionary matching ratio -> length of matched words / total domain length 
17. Number of subdomains
18. Well-known tld 
19. Contains www
"""
import os
import math
import nltk
import glob
import json
import tldextract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Custom modules
from features import Features
from dataset_paths import DatasetPaths
from extract_features import (
    extract_features_from_sld,
    extract_features_from_concat,
    extract_features_averaged,
    extract_features_from_max,
    extract_features_4,
    ) 

from nltk.corpus import words
from n_grams import N_grams
from known_tld import KnownTLD
from sklearn.decomposition import PCA

def get_multiclass_labels(dir_path:str) -> dict:
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

def save_multiclass_labels_to_json(labels: dict, filename: str):
    """
    Function saves generated labels for further
    processing as part of the DGA classification
    into pickle file.

    Args:
        labels (dict): labels that will be saved
        filename (str): name of the pickle file
    """
    with open(filename, "w") as f:
        json.dump(labels, f)

def load_multiclass_labels_from_json(filename: str) -> dict:
    """
    Function saves generated labels for further
    processing as part of the DGA classification
    into pickle file.

    Args:
        labels (dict): labels that will be saved
        filename (str): name of the pickle file
    """
    with open(filename, "r") as f:
        dga_family_labels = json.load(f)
        return dga_family_labels
def read_dga_df(dataset_dir:str, column_name:str) -> pd.DataFrame:
    """Function read all csv files from specified directory
    into one dataframe

    Args:
        dataset_dir (str): Directory with csv files.
        column_name (str): Name for the column, where the domain name
                           will be stored.

    Returns:
        pd.DataFrame: Concatenated dataframe of DGA families
    """
    # Retrieve all CSV file paths in the directory
    csv_files = glob.glob(dataset_dir + "*.csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, names=column_name)
        dfs.append(df)

    dga_df = pd.concat(dfs, ignore_index=True)
    return dga_df 

def read_datasets_for_binary_clf(dataset_paths: DatasetPaths, features: Features) -> pd.DataFrame:
    dga_df = read_dga_df(dataset_paths.get_dga_binary_clf_dataset_path(),
                         column_name=["domain"])
    nondga_df = pd.read_csv(dataset_paths.get_nondga_binary_clf_dataset_paht(),
                            names=["domain_name","label"],
                            )
    nondga_df.rename(columns={'domain_name': 'domain'}, inplace=True)
    nondga_df.drop('label', axis=1, inplace=True)
    
    dga_df.drop_duplicates()
    nondga_df.drop_duplicates()
    # Label datasets
    dga_label = features.get_binary_label_column()
    dga_df[dga_label] = 1
    nondga_df[dga_label] = 0
    
    final_df = pd.concat([dga_df, nondga_df], ignore_index=True)
    return final_df

def read_datasets_for_multiclass_clf(dataset_paths: DatasetPaths, features: Features) -> pd.DataFrame:
    # def read_and_label_dataset(dataset_dir:str, label_column:str):
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

    domain_col = features.get_domain_column()
    label_col = features.get_multiclass_label_column()

    dga_dataset_dir = dataset_paths.get_dga_multiclass_clf_dataset_dir()
    family_labels = get_multiclass_labels(dga_dataset_dir)
    file_list = glob.glob(dga_dataset_dir + prefix + "*" + suffix)
    
    for file in file_list:
        # 3. Assign the labels
        filename = os.path.basename(file)
        family = filename[len(prefix):-len(suffix)] # Remove prefix and suffix from filename
        label = family_labels[family] 
        df = pd.read_csv(file, names=[domain_col])
        df.drop_duplicates()
        df[label_col] = label                
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) #concatenated dataframe

def extract_features_from_df(domains_df:pd.DataFrame, features:Features) -> pd.DataFrame:
    SLD_EXTRACT = 0
    CONCAT_SUBDOMAINS = 1
    MEAN_FROM_SUBDOMAIN_FEATURES = 2
    MAX_FROM_SUBDOMAIN_FEATURES = 3
    
    type_of_subdomain_processing = CONCAT_SUBDOMAINS
    feature_columns = features.get_feature_columns()
    domain_name_column = features.get_domain_column()
    
    known_tlds = KnownTLD().get_tlds()
    ngrams = N_grams()
    
    if type_of_subdomain_processing == SLD_EXTRACT: 
        domains_df[feature_columns] = domains_df[domain_name_column].apply(lambda x: extract_features_from_sld(x, known_tlds, ngrams)).apply(pd.Series)
    elif type_of_subdomain_processing == CONCAT_SUBDOMAINS:  
        domains_df[feature_columns] = domains_df[domain_name_column].apply(lambda x: extract_features_from_concat(x, known_tlds, ngrams)).apply(pd.Series)
    elif type_of_subdomain_processing == MEAN_FROM_SUBDOMAIN_FEATURES:
        domains_df[feature_columns] = domains_df[domain_name_column].apply(lambda x: extract_features_averaged(x, known_tlds, ngrams)).apply(pd.Series)
    elif type_of_subdomain_processing == MAX_FROM_SUBDOMAIN_FEATURES: 
        domains_df[feature_columns] = domains_df[domain_name_column].apply(lambda x: extract_features_from_max(x, known_tlds, ngrams)).apply(pd.Series)
    else: # Vector projecting -> logically not appliable
        domains_df[feature_columns] = domains_df[domain_name_column].apply(lambda x: extract_features_4(x, known_tlds, ngrams)).apply(pd.Series)
        
    return domains_df

def extract_features_for_binary_clf(filename:str=None):
    dataset_paths = DatasetPaths()
    features = Features()
    domains_df = read_datasets_for_binary_clf(dataset_paths, features)
    
    extracted_features_df = extract_features_from_df(domains_df, features)
    if filename:
        extracted_features_df.to_csv(filename, index=False)
    return extracted_features_df
    
def extract_features_for_multiclass_clf(filename:str=None):
    dataset_paths = DatasetPaths()
    features = Features()
    domains_df = read_datasets_for_multiclass_clf(dataset_paths, features)
    extracted_features_df = extract_features_from_df(domains_df, features)
    if filename:
        extracted_features_df.to_csv(filename, index=False)
    return extracted_features_df

def main():
    # determine which type of features are going to be calculated
    binary_features_savedir = DatasetPaths().get_binary_clf_features_savepath()
    multiclass_features_savedir = DatasetPaths().get_multiclass_clf_features_savepath()
    extract_features_for_binary_clf(binary_features_savedir + 
                                    "dga_binary_clf_features.csv")
    # extract_features_for_multiclass_clf(multiclass_features_savedir + 
    #                                     "dga_multiclass_clf_features.csv")
if __name__=="__main__":
    main()
    
    
    
    
    
    
    