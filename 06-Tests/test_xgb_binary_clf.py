#!/usr/bin/env python3
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

# 1. Load binary model
# 2. Evalutation metrics
# 3. Graphs
# 3.1. Confusion matrix
# 3.2. ROC AUC model curve
# 3.3. Feature importances

import sys
import pickle 
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    )
from sklearn.model_selection import train_test_split, cross_val_score
# Import custom modules

sys.path.append("./03-Models-training/")
sys.path.append("./02-Data-preprocessing/03-Feature-extraction/")

from features import Features
from model_paths import ModelPaths
import train_xgb_binary_clf as binaryclf

def load_binary_model() -> xgb.XGBClassifier:
    model_dir = ModelPaths().get_xgb_binary_clf_model_path()
    model_name = "01_xgb_binary_clf.pkl"
    model_filename = model_dir + model_name
    
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
        return model
    
    
def evaluate_model(model, X, y, X_test, y_test):

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    # scores = cross_val_score(model, X, y, cv=5)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print('ROC AUC:', roc_auc)
    # print('Scores:', scores.mean())

def visualize_confusion_matrix(model, X_test, y_test):
    # Plot the confusion matrix as a heatmap
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    sns.set(font_scale=1.5)
    sns.heatmap(cm,
                annot=True,
                cmap='Greens',
                fmt='g',
                xticklabels=['Predicted Non-DGA', 'Predicted DGA'],
                yticklabels=['Actual Non-DGA', 'Actual DGA']
                )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.savefig("xgb_binary_clf_cm.png")
    plt.show()
    pass

def visualize_feature_importance(model:xgb.XGBClassifier):
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, color="green")
    # plt.savefig("xgb_binary_clf_feature_importance.png")
    plt.show()

def visualize_roc_auc_curve(model, X_test, y_test):
    # Predict probabilities on the test set
    y_pred_proba = model.predict_proba(X_test)[:,1]
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkgreen', lw=lw, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig("xgb_binary_clf_roc_curve.png")
    plt.show()
    pass
def visualize_correlation_matrix(corr_matrix):
    # create a heatmap of the correlation matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                linewidths=0.5,
                ax=ax,
                annot_kws={"size": 13},
                vmax=1, vmin=-1, center=0,
                square=True
                )

    # increase the font size of the labels and the correlation values
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.show()
    
def main():
    # 1. Read features df
    features_df = binaryclf.get_features_df()
    # 2. Train test split
    
     # Split the data
    feature_columns = Features().get_feature_columns()
    class_labels = Features().get_binary_label_column()
    
    X = features_df[feature_columns]
    y = features_df[class_labels]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=42)
    corr_matrix = X.corr().round(2)
    visualize_correlation_matrix(corr_matrix)
    model = load_binary_model()
    evaluate_model(model, X, y, X_test, y_test)
    visualize_confusion_matrix(model, X_test, y_test)
    visualize_roc_auc_curve(model, X_test, y_test)
    visualize_feature_importance(model)
if __name__=="__main__":
    main()