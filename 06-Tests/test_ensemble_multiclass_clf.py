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
This scripts serves as an module for performance evaluation of the XGBoost multiclass
classification
"""

# 1. Load multiclass model
# 2. Evalutation metrics
# 3. Graphs
# 3.1. Confusion matrix
# 3.2. ROC AUC model curve
# 3.3. Feature importances

import sys
import time 
import pickle
sys.path.append("./05-Classifiers")
sys.path.append("./02-Data-preprocessing/03-Feature-extraction")
sys.path.append("./03-Models-training")
sys.path.append("./03-Models-training/01-Ensemble-train")

from collections import Counter
from features import Features
from extract_features_to_df import get_multiclass_labels
from dataset_paths import DatasetPaths

from ensemble_model import  EnsembleClassifier
from model_paths import ModelPaths
import train_xgb_multiclass_clf as multicls


from dga_classif_features import get_labels
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    RocCurveDisplay,
)

from yellowbrick.classifier import ClassPredictionError
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

def load_multiclass_model() -> xgb.XGBClassifier:
    model_dir = ModelPaths().get_xgb_multiclass_clf_model_path()
    model_name = "01_dga_xgb_multiclass_clf.pkl"
    model_filename = model_dir + model_name
    
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
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

def evaluate_model(model, X, y, X_test, y_test):

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
    cohen_kappa = cohen_kappa_score(y_test, y_pred, weights="linear")
    # scores = cross_val_score(model, X, y, cv=5)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print("Cohen's Kappa:", cohen_kappa)
    
    # print('Scores:', scores.mean())

def visualize_class_prediction_errors(model, X_train, y_train, X_test, y_test):
    family_labels_dir = DatasetPaths().get_dga_multiclass_clf_dataset_dir()
    dga_family_labels_dict = get_multiclass_labels(family_labels_dir)
    dga_family_labels = [dga_family[:-4] for dga_family in dga_family_labels_dict.keys()]
    
    visualizer = ClassPredictionError(model,
                                      classes=dga_family_labels)

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.show()

def visualize_confusion_matrix(model, X_test, y_test):
    # Plot the confusion matrix as a heatmap
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{execution_time // 60}m{execution_time%60}s")
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    total_per_class = np.sum(cm, axis=1)
    # calculate the number of misclassified instances per class
    misclassified_per_class = total_per_class - np.diag(cm)
    # calculate the misclassification rate per class
    misclassification_rate_per_class = misclassified_per_class / total_per_class
    # print the misclassification rate per class
    number_of_classes = cm.shape[0]
    for i in range(number_of_classes):
        print(f"Class {i}: {misclassification_rate_per_class[i]:.2%}")
    family_labels_dir = DatasetPaths().get_dga_multiclass_clf_dataset_dir()
    labels = get_multiclass_labels(family_labels_dir)
    cm_normalized = np.round(cm/np.sum(cm, axis=1).reshape(-1,1),2)
    sns.heatmap(cm_normalized,
                cmap="Greens",
                annot=False,
                xticklabels=[ label[:-len("_dga")] for label in labels.keys() ],
                yticklabels=[ label[:-len("_dga")] for label in labels.keys() ])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    pass

def visualize_feature_importance(model:xgb.XGBClassifier):
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, color="green")
    
    plt.show()

def visualize_classification_report(model:xgb.XGBClassifier,
                                    X_test,
                                    y_test,
                                    start_class=None,
                                    num_class=None):
    
    # Generate classification report
    y_pred = model.predict(X_test)
    family_labels_dir = DatasetPaths().get_dga_multiclass_clf_dataset_dir()
    dga_family_labels_dict = get_multiclass_labels(family_labels_dir)
    dga_family_labels = [dga_family[:-4] for dga_family in dga_family_labels_dict.keys()]
    report = classification_report(y_test,
                                   y_pred,
                                   zero_division=1,
                                #    labels=dga_family_labels,
                                   output_dict=True,
                                   )
    report.pop("accuracy")
    report.pop("macro avg")
    report.pop("weighted avg")
    dga_family_labels = [ label + " ("+str(support["support"])+")"
                         for label, support in zip(dga_family_labels, report.values())]
    family_labels_dir = DatasetPaths().get_dga_multiclass_clf_dataset_dir()
    dga_family_labels_dict = get_multiclass_labels(family_labels_dir)
   
    num_classes = 93
    chunk_size = 20
    sns.set(font_scale=1.3)
    for i in range(0, num_classes, chunk_size):
        start_class = i
        upper_limit = min(i + chunk_size, num_classes)
        num_class = upper_limit - start_class
        fig = plt.figure(figsize=(8, 12))
        sns.heatmap(pd.DataFrame(report).iloc[:-1, start_class:].T.head(n=num_class),
                    annot=True,
                    yticklabels=dga_family_labels[start_class:start_class+num_class],
                    cmap="Blues",
                    linewidths=1,
                    square=True,
                    fmt=".2f"
                    # annot_kws={"size": 14},
                    )
        
        # Show the plot
        plt.title(f"Families {start_class}:{start_class+num_class}")
        plt.tight_layout()
        # plt.savefig(f"{start_class}_{start_class+num_class}_xgb_multiclass_scores.png")
        plt.show()
def main():
    # 1. Read features df
    features_df = multicls.get_features_df()
    # 2. Train test split
    
    # Split the data
    feature_columns = Features().get_feature_columns()
    class_labels = Features().get_multiclass_label_column()
    
    X = features_df[feature_columns]
    y = features_df[class_labels]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.5,
                                                        stratify=y,
                                                        random_state=42)
    
    model = EnsembleClassifier()
    visualize_classification_report(model, X_test, y_test)
    evaluate_model(model, X, y, X_test, y_test)
    visualize_confusion_matrix(model, X_test, y_test)
    # visualize_feature_importance(model)
    num_classes = 93
    chunk_size = 20
    save_graph = False
    if save_graph:
        for i in range(0, num_classes, chunk_size):
            start_class = i
            num_class = min(i + chunk_size, num_classes)
            visualize_classification_report(model, X_test, y_test,
                                            start_class=start_class, 
                                            num_class=num_class - start_class)
if __name__=="__main__":
    main()