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
The script creates 3 models, which all cooperate in multiclass DGA
classification.
The first model the first model serves to filter out poorly represented
DGA families from strongly represented DGA families in data distribution
Second and Third model serves for the final classification into families.
"""
import os
import sys
sys.path.append("./02-Data-preprocessing/03-Feature-extraction")
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from n_grams import N_grams
from known_tld import KnownTLD
from extract_features import extract_features_from_concat
from dataset_paths import DatasetPaths
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

def read_dataset(file_csv:str):
    """Function reads the dataset from csv file
    and returns the pandas DataFrame

    Args:
        file_csv (str): Filepath to the csv file
    """
    df = pd.read_csv(file_csv)

def compute_features(dataset_df:pd.DataFrame, feature_columns:list) -> pd.DataFrame:
    """Function extracts features from given dataset
    and returns dataframe with computed features

    Args:
        dataset_df (pd.DataFrame): Dataframe with DGA samples
        feature_columns (list) : List of features which are going to be extracted
    Returns:
        pd.DataFrame: Dataframe with DGA samples and extracted features
    """
    known_subdomain_path = DatasetPaths().get_known_tlds_path()
    tlds = KnownTLD(known_subdomain_path)
    known_tlds = tlds.get_tlds()
    
    ngram_dir = DatasetPaths().get_ngrams_dir()
    dga_ngram_csv = ngram_dir + "dga-ngram.csv"
    nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
    ngrams = N_grams(dga_ngram_csv=dga_ngram_csv, nondga_ngram_csv=nondga_ngram_csv)
    
    dataset_df[feature_columns] = (dataset_df["domain"]
                                   .apply(lambda x: extract_features_from_concat(x, known_tlds, ngrams))
                                   .apply(pd.Series))
    return dataset_df

def split_the_dataset(features_df:pd.DataFrame, 
                      feature_columns:list, 
                      class_name:str,
                      split_ratio:float,
                      stratify:bool=False):
    """Function splits the data into 2 groups according to the 
    specified ratio.

    Args:
        features_df (pd.DataFrame): Data with computed features
        feature_columns (list): Names of the features
        class_name (str): The name of the column by which we will classify
        split_ratio (float): The ratio based on which the data will be divided
        stratify (bool): To ensure that the target variable's class distribution
                         is maintained in both the training and testing datasets
                         during the process of splitting data into training and
                         testing sets.
    """
    # Split the data
    X = features_df[feature_columns]
    y = features_df[class_name]
    if stratify:
        return train_test_split(X, y, test_size=split_ratio, stratify=y, random_state=42)
    else:
        return train_test_split(X, y, test_size=split_ratio, random_state=42)
        
def create_model(X_train:pd.DataFrame, y_train:pd.Series):
    """Function trains and creates the xgboost model on given dataset

    Args:
        features_df (pd.DataFrame): Dataframe with features
        feature_columns (list): Feature names
        class_name (str): The name of the column by which we will classify
    """
    # Train the XGBoost model
    params = {'objective': 'binary:logistic',
              'max_depth': 6, 
              'learning_rate': 0.6, 
              'n_estimators': 100}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def create_model_for_poorly_represented_classes(
    X_train:pd.DataFrame, 
    y_train:pd.Series,
    num_classes:int):
    """Function trains and creates the xgboost model on given dataset
    for multiclass classification on classes with poor distribution

    Args:
        X_train (pd.DataFrame): Feature columns
        y_train (pd.Series): DGA Families (classes)
        num_classes (int): Number of classes

    Returns:
        xgb.XGBClassifier: trained model
    """
    # Train the XGBoost model
    params = {'objective': 'multi:softmax',
              'num_class': num_classes,
              'max_depth': 3, 'learning_rate': 0.5,
              'n_estimators': 100}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_binary_model(model:xgb.XGBClassifier,
                   X_test:pd.DataFrame,
                   y_test:pd.Series,
                   X:pd.DataFrame,
                   y:pd.Series):
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5)
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print('ROC AUC:', roc_auc)
    print('Scores:', scores.mean())

def evaluate_multiclass_model(model:xgb.XGBClassifier,
                   X_test:pd.DataFrame,
                   y_test:pd.Series
                   ):
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') # Specify multi-class strategy as 'ovr'
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')

    # Print the evaluation metrics
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: ", confusion)
    print("Classification Report: ")
    print(classification)
    print("F1 Score: ", f1)
    print("ROC AUC Score: ", roc_auc)
    print("Precision: ", precision)
    print("Recall: ", recall)

def train_preclassifcation_model():
    # 1. Read 2 datasets from csv files
    datasets_dir = DatasetPaths().get_median_split_dataset_dir()
    dga_samples_below_median = datasets_dir + "01-classes-below-median.csv"
    dga_samples_above_median = datasets_dir + "01-classes-above-median.csv"
    
    sparse_dga_samples_df = pd.read_csv(dga_samples_below_median, names=["domain"], skiprows=1)
    dense_dga_samples_df = pd.read_csv(dga_samples_above_median, names=["domain"], skiprows=1)
    
    # 2. Add labels to them
    sparse_dga_samples_df["median_split"] = 0
    dense_dga_samples_df["median_split"] = 1
    
    dataset_df = pd.concat([sparse_dga_samples_df, dense_dga_samples_df], ignore_index=True)
    # 3. Extract features
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
    
    dataset_with_features_df = compute_features(dataset_df, feature_columns)
    # 4. Split the data
    X_train, X_test, y_train, y_test = split_the_dataset(dataset_with_features_df,
                                                         feature_columns,
                                                         class_name="median_split",
                                                         split_ratio=0.4)
    # 5. Create the model
    preclassification_model = create_model(X_train, y_train)
    
    # 6. Evalute the model
    X = dataset_with_features_df[feature_columns]
    y = dataset_with_features_df["median_split"]
    evaluate_binary_model(preclassification_model, X_test, y_test, X=X, y=y)
    
    # 7. Save the model
    model_path = "./03-Models-training/01-Ensemble-train/01-Models/"
    model_name = "01-binary-clf.pkl"
    # if not os.path.exists(model_path + model_name):
    #     save_model(preclassification_model, model_path + model_name)
        
        
def get_class_labels(dga_samples_df:pd.DataFrame, class_column:str) -> dict:
    """Function assigns labels to classes stated in the dataframe.
    

    Args:
        dga_samples_df (pd.DataFrame): Dataframe containing DGA samples

    Returns:
        dict: Dictionary where key is the family name and value
              is the label (int number)
    """
    # Get all families in the dataset
    dga_families = dga_samples_df[class_column].unique() # Sort in alphabetical order
    dga_families.sort()
    
    # Create labels for Families
    return { family:idx for idx, family in enumerate(dga_families) }

def save_labels(dga_labels:dict, json_out:str):
    """Function stores labels for dga_families
    used for classification

    Args:
        dga_labels (dict): Key -> dga_family | Value -> int 0 to n
        json_out (str): dictionary stored in json file
    """
    # Save dictionary to a JSON file
    with open(json_out, 'w') as f:
        json.dump(dga_labels, f)

def load_labels(json_label_file:str) -> dict:
    """Function loads dga labels into python dictionary

    Args:
        json_label_file (str): Path to the file

    Returns:
        dict: Labels for dga families used within classification
    """
        # Load JSON file into a Python dictionary
    with open(json_label_file, 'r') as f:
        dga_labels = json.load(f)
    
    if dga_labels:
        return dga_labels        

def save_model(model:xgb.XGBClassifier, model_path:str):
    """Function stores an XGBoost model in pickle format
    in the path speicifed by model_path.

    Args:
        model (xgb.XGBClassifier): trained model 
        model_path (str): path where model will be stored
    """
    # Check if the model exists
    print("Saving the model")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
def load_model(model_path:str):
    """Function stores an XGBoost model in pickle format
    in the path speicifed by model_path.

    Args:
        model (xgb.XGBClassifier): trained model 
        model_path (str): path where model will be stored
    """
    print("Model exists")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def train_multiclass_dga_model(dataset_path:str, model_path:str=None, labels_path:str=None):
    """Trains multiclass classification model
    for DGA families

    Args:
        dataset_path (str): Path to the dga families
    """
    # 1. Read the dataset
    columns=['domain','dga_family']
    dga_samples_df = pd.read_csv(dataset_path, names=columns)
    
    
    # 2. Add labels to dataframe
    family_labels = get_class_labels(dga_samples_df, class_column="dga_family")
    dga_samples_df["dga_family_label"] = dga_samples_df["dga_family"].map(family_labels)
    
    # 3. Extract features
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
    dataset_with_features_df = compute_features(dga_samples_df, feature_columns)

    # 4. Split the data
    X_train, X_test, y_train, y_test = split_the_dataset(dataset_with_features_df,
                                                         feature_columns,
                                                         class_name="dga_family_label",
                                                         split_ratio=0.2,
                                                         stratify=True)
    
    # 5. Train the XGBoost model
    model = create_model_for_poorly_represented_classes(X_train,
                                                y_train,
                                                num_classes=len(family_labels))

    # 6. Evaluate the model
    evaluate_multiclass_model(model,
                              X_test,
                              y_test)
    # 7. Save the model and labels
    if model_path:
        save_model(model, model_path)
    if labels_path:
        save_labels(family_labels, labels_path)
        
def main():
    datasets_dir = DatasetPaths().get_median_split_dataset_dir()
    train_preclassifcation_model()
    
    poorly_represented_dga = datasets_dir + "02-poorly-represented-dga.csv"
    model_path = "./03-Models-training/01-Ensemble-train"
    model_name = "/01-Models/02-below-median-clf.pkl"
    labels_name = "/02-Labels/02-below-median-labels.json"
    
    train_multiclass_dga_model(poorly_represented_dga, model_path=model_path + model_name, labels_path=model_path + labels_name)
    
    densly_represented_dga = datasets_dir + "02-densly-represented-dga.csv"
    model_path = "./03-Models-training/01-Ensemble-train"
    model_name = "/01-Models/03-above-median-clf.pkl"
    labels_name = "/02-Labels/03-above-median-labels.json"
    train_multiclass_dga_model(densly_represented_dga, model_path=model_path + model_name, labels_path=model_path + labels_name)

if __name__=="__main__":
    main()