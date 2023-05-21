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
Module implements object which will hold all the paths
necessary for reading the datasets for binary and multiclass
classifiers training.
"""

import os
import sys

class ModelPaths:
    # 1. Init the modelpaths
    
    def __init__(self) -> None:
        self.__project_root_dir = "DGA-Classifier"
        self.__verify_root_directory()
        # 1.1 Paths for reading models
        self.__dga_xgb_binary_clf_path = "/mnt/d/VUT/Bachelor-thesis/" \
                                         "05-Github/DGA-Classifier/" \
                                         "04-Models/binary_classification/"
        self.__dga_xgb_multiclass_clf_path = "/mnt/d/VUT/Bachelor-thesis/" \
                                             "05-Github/DGA-Classifier/" \
                                             "04-Models/multiclass_classification/01-Models/"
        self.__xgb_binary_clf = "./04-Models/binary_classification/01_xgb_binary_clf.pkl"
        self.__xgb_multiclass_clf = "./04-Models/multiclass_classification/01-Models/01_dga_xgb_multiclass_clf.pkl"
        self.__regex_clf = "./04-Models/regex_classification/regexes.txt"
        self.__stat_clf = "./04-Models/statistical_classification/classif_matrix"
                                                     
        # 1.2 Paths for saving models
        self.__dga_xgb_binary_clf_savepath = "/mnt/d/VUT/Bachelor-thesis/" \
                                             "05-Github/DGA-Classifier/" \
                                             "04-Models/binary_classification/"
        self.__dga_xgb_multiclass_clf_savepath = "/mnt/d/VUT/Bachelor-thesis/" \
                                             "05-Github/DGA-Classifier/" \
                                             "04-Models/multiclass_classification/01-Models/"
        # 1.3 Paths for multiclass labels
        self.__dga_xgb_multiclass_clf_labels_path = "./04-Models/multiclass_classification/02-Labels/"
        
    def __verify_root_directory(self):
        cwd = os.getcwd()
        dirname = os.path.basename(cwd)
        if dirname != self.__project_root_dir:
            print("Run the script from the project root directory: ", file=sys.stderr)
            exit(1)
            
    #Setter methods
    def set_xgb_binary_clf_model_path(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_xgb_binary_clf_path = filename

        
    def set_xgb_multiclass_clf_model_path(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_xgb_multiclass_clf_path = filename
    
    def set_xgb_binary_clf_model_savepath(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_xgb_binary_clf_path = filename

        
    def set_xgb_multiclass_clf_model_savepath(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_xgb_multiclass_clf_path = filename
        
    def set_xgb_multiclass_clf_labels_path(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_xgb_multiclass_clf_path = filename
    
    # Getter methods
    def get_xgb_binary_clf_model_path(self) -> str:
        return self.__dga_xgb_binary_clf_path

    def get_xgb_multiclass_clf_model_path(self) -> str:
        return self.__dga_xgb_multiclass_clf_path
    
    def get_xgb_binary_clf_model_savepath(self) -> str:
        return self.__dga_xgb_binary_clf_savepath

    def get_xgb_multiclass_clf_model_savepath(self) -> str:
        return self.__dga_xgb_multiclass_clf_savepath
    
    def get_xgb_multiclass_clf_labels_path(self) -> str:
        return self.__dga_xgb_multiclass_clf_labels_path
    
    def get_binary_clf(self) -> str:
        return self.__xgb_binary_clf
    def get_multiclass_clf(self) -> str:
        return self.__xgb_multiclass_clf
    def get_stat_clf(self) -> str:
        return self.__stat_clf
    def get_regex_clf(self) -> str:
        return self.__regex_clf
