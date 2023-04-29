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
This scripts serves as an moduled for performance evaluation of the XGBoost multiclass
classification
"""

import sys
import pickle
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "01-Classifier")
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "02-Dataset-preprocessing/04-Feature-extraction")
sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"\
                "02-Dataset-preprocessing/05-Multiclass-clf")
from dga_classif_features import get_labels
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import xgboost as xgb

def read_df_with_features(path:str):
    df = pd.read_csv(path)
    return df

def load_model(model_path:str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if model:
        return model
   
   
def plot_roc_curve_for_specific_class(class_id:int, y_onehot_test, y_pred):
    class_of_interest = "bamital"
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_pred[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
    plt.legend()
    plt.show()
    
def plot_roc_curve_for_multiclass(y_onehot_test, y_pred):
    RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_pred.ravel(),
    name="micro-average OvR",
    color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
    plt.legend()
    plt.show()

def main():
    dir_path= "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/"
    dataset_name = "dga_multiclass_detection_features.csv"
    model_path = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/multiclass_classification/"
    model_name = "01_multiclass_model_subdomain.pkl"
              
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
    
    
    dataset_with_features_df = read_df_with_features(dataset_name)
    print(dataset_with_features_df)
    
    X = dataset_with_features_df[feature_columns]
    y = dataset_with_features_df["dga_family"]
    
    (
    X_train,
    X_test,
    y_train,
    y_test,
    ) = train_test_split(X, y, test_size=0.5, stratify=y, shuffle=True)
    
    model = load_model(model_path + model_name)
    
    y_pred = model.predict_proba(X_test)
    print(y_pred)
    
    # Binarize the target    
    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)
    
    print(y_onehot_test.shape)
    print(y_onehot_test)

    #import matplotlib.pyplot as plt
    # xgb.plot_importance(model, max_num_features=20)
    # plt.show()
    
    class_id = 2
    # plot_roc_curve_for_multiclass(y_onehot_test, y_pred)

    
    # assume y_true and y_pred are the true and predicted labels, respectively
    # for the test set of your XGBoost model
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot the normalized confusion matrix using seaborn
    plt.figure(figsize=(80, 80))
    sns.set(font_scale=3.3)
    # sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    
    # plt.figure(figsize=(20,15))
    # plt.imshow(confusion_matrix, interpolation='none')
    # plt.colorbar()
    # plt.xticks(ticks,fontsize=6)
    # plt.yticks(ticks,fontsize=6)
    # plt.grid(True)
    # plt.show()
    #SET ticks
    dir_path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_datasets/02-Proportion-pick/"
    labels = get_labels(dir_path)
    print(labels)
    
    number_of_classes = cm.shape[0]
    ticks=np.linspace(0, number_of_classes - 1,num=number_of_classes)
    
    cm_normalized = np.round(cm_norm/np.sum(cm_norm, axis=1).reshape(-1,1),2)
    sns.heatmap(cm_normalized,
                cmap="Greens",
                annot=False,
                xticklabels=[ label[:-len("_dga")] for label in labels.keys() ],
                yticklabels=[ label[:-len("_dga")] for label in labels.keys() ],
                fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix_xgboost_multiclass.png")
    plt.show()
if __name__=="__main__":
    main()