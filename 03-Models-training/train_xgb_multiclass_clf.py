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
Modul is created for training the XGBoost binary classifier.
Training process:
1. Get features for training.
Features for training could be obtained by:
a) Calculating from dataset
b) Or if they are already calculated, by loading from csv file

2. Set XGBoost hyperparameters -> by previos experimenting we set the parameters
which provided the best results

3. Evaluate the model
4. Save the model
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
     accuracy_score,
     precision_score,
     recall_score,
     f1_score,
     roc_auc_score,
    )

# Import custom modules
sys.path.append("./02-Data-preprocessing/03-Feature-extraction")

from features import Features
from dataset_paths import DatasetPaths
import extract_features_to_df as efr
from model_paths import ModelPaths

def get_features_df() -> pd.DataFrame:
    """Function returns the dataset with features
    neccessary for model training.
    If the dataset with computed features exists it's loaded
    from the directory.
    If the dataset with features does not exists, the features
    are extracted from dataset containing only domain names

    Returns:
        pd.DataFrame: Dataframe containing extracted features from
        domains, and domain labels, neccessary for classification
    """
    multiclass_clf_features_df_filename = DatasetPaths().get_multiclass_clf_features_path()
    if os.path.isfile(multiclass_clf_features_df_filename):
        # Load features df
        features_df = pd.read_csv(multiclass_clf_features_df_filename)
        return features_df
    else:
        features_df = efr.extract_features_for_multiclass_clf()
        return features_df

def split_the_dataset(features_df:pd.DataFrame, 
                      split_ratio:float,
                      stratify:bool=False):
    """Function splits the data into 2 groups according to the 
    specified ratio.

    Args:
        features_df (pd.DataFrame): Data with computed features
        split_ratio (float): The ratio based on which the data will be divided
        stratify (bool): To ensure that the target variable's class distribution
                         is maintained in both the training and testing datasets
                         during the process of splitting data into training and
                         testing sets.
    """
    # Split the data
    feature_columns = Features().get_feature_columns()
    class_labels = Features().get_multiclass_label_column()
    X = features_df[feature_columns]
    y = features_df[class_labels]
    if stratify:
        return train_test_split(X, y, test_size=split_ratio, stratify=y, random_state=42)
    else:
        return train_test_split(X, y, test_size=split_ratio, random_state=42)

def create_model(X_train:pd.DataFrame, y_train:pd.Series, class_weights:pd.Series=None):
    """Function trains and creates the xgboost model on given dataset

    Args:
        features_df (pd.DataFrame): Dataframe with features
        feature_columns (list): Feature names
        class_name (str): The name of the column by which we will classify
    """
    params = {'objective': 'multi:softmax',
              'num_class': 93,
            #   "weights":class_weights,
              'max_depth': 6,
              'learning_rate': 0.06, 
              'n_estimators': 150,
              }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def save_model(model:xgb.XGBClassifier, filename:str=None) -> None:
    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_savepath = ModelPaths().get_xgb_multiclass_clf_model_savepath()
        model_name = "01_dga_xgb_multiclass_clf.pkl"
        with open(model_savepath + model_name, 'wb') as f:
            pickle.dump(model, f)

def evaluate_model(model, X_test, y_test):
     
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    # scores = cross_val_score(model, X, y, cv=5)
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print('ROC AUC score:', roc_auc)
    # print('Scores:', scores.mean())

def train_xgb_multiclass_clf():
    features_df = get_features_df()
    print(features_df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_the_dataset(features_df,
                                                         split_ratio=0.2,
                                                         stratify=True)
    # Train the XGBoost model
    model = create_model(X_train, y_train)

    # 6. Evaluate the model
    # evaluate_model(model, X_test, y_test)
    # # 7. Save the model and labels
    model_savepath = ModelPaths().get_xgb_multiclass_clf_model_savepath()
    model_name = "02_dga_xgb_multiclass_clf.pkl"
    save_model(model, model_savepath + model_name)
    # if model_path:
    #     save_model(model, model_path)
    # if labels_path:
    #     save_labels(family_labels, labels_path)
    # pass

def main():
    # Get features dataset
    # 1. If not saved then compute online
    # 2. Oversampling/Undersampling
    # 3. Create XGBoost model
    # 4. Evaluate the model
    # 5. Save the model
    # 6. Tune hyperparameters
    train_xgb_multiclass_clf()
if __name__=="__main__":
    main()