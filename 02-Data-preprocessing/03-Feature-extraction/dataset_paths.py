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

class DatasetPaths:
    # 1. Init the datasetpaths
    def __init__(self) -> None:
        self.__project_root_dir = "DGA-Classifier"
        self.__verify_root_directory()
        # 1.1 Paths for reading datasets
        self.__dga_binary_clf_dataset_path = "./01-Data/02-Preprocessed-data/"\
                                             "DGA/01-Proportion-pick/"
        self.__dga_multiclass_clf_dataset_dir_path = "./01-Data/02-Preprocessed-data/"\
                                                     "DGA/02-Proportion-pick/"
        self.__nondga_binary_clf_dataset_path = "./01-Data/02-Preprocessed-data/Non-DGA/"\
                                                "cisco-umbrella-nes-fit-verified.csv"
        self.__dga_binary_clf_features_path = "./01-Data/03-Extracted-features-data/" \
                                              "dga_binary_clf_features.csv"
        self.__dga_multiclass_clf_features_path = "./01-Data/03-Extracted-features-data/" \
                                                  "dga_multiclass_clf_features.csv"
        self.__stat_clf_labels_dir = "./01-Data/02-Preprocessed-data/"\
                                     "DGA/01-Proportion-pick/"
        # N-grams
        self.__dga_extract_ngram_dir = "./01-Data/02-Preprocessed-data/"\
                                       "DGA/01-Proportion-pick"
        self.__dga_extract_ngram_full_dir = "./01-Data/01-Raw-data/dga_archive_full/"
        self.__non_dga_extract_ngram_dir = "./01-Data/02-Preprocessed-data/"\
                                           "Non-DGA/"
        self.__ngrams_dir = "./04-Models/ngrams/"
        # Known-TLD
        self.__known_tlds_path = "./01-Data/01-Raw-data/public_suffix_list.dat.txt"
        
        # 1.2 Path for saving datasets with extracted features
        self.__binary_clf_dataset_with_features_savepath = "./01-Data/03-Extracted-features-data/"
        self.__multiclass_clf_dataset_with_features_savepath = "./01-Data/03-Extracted-features-data/"
        self.__multiclass_clf_class_labels_path = "./01-Data/04-Labels/"
        
    def __verify_root_directory(self):
        cwd = os.getcwd()
        dirname = os.path.basename(cwd)
        if dirname != self.__project_root_dir:
            print("Run the script from the project root directory: ", file=sys.stderr)
            exit(1)
            
    # Setter methods
    def set_dga_binary_clf_load_path(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_binary_clf_dataset_path = filename
    
    def set_dga_multiclass_clf_load_dir(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_multiclass_clf_dataset_dir_path = filename
        
    def set_nondga_binary_clf_load_path(self, filename:str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__nondga_binary_clf_dataset_path = filename
    
    def set_binary_clf_features_savepath(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__binary_clf_dataset_with_features_savepath = filename
    
    def set_multiclass_clf_features_savepath(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__multiclass_clf_dataset_with_features_savepath = filename
    
    def set_binary_clf_features_path(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_binary_clf_features_path = filename
        
    def set_multiclass_clf_features_path(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__dga_multiclass_clf_features_path = filename
    
    def set_multiclass_clf_class_labels_path(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        self.__multiclass_clf_class_labels_path = filename
        
    # Getter methods
    def get_dga_binary_clf_dataset_path(self) -> str:
        return self.__dga_binary_clf_dataset_path
    
    def get_dga_multiclass_clf_dataset_dir(self) -> str:
        return self.__dga_multiclass_clf_dataset_dir_path
    
    def get_nondga_binary_clf_dataset_paht(self) -> str:
        return self.__nondga_binary_clf_dataset_path
    
    def get_binary_clf_features_savepath(self) -> str:
        return self.__binary_clf_dataset_with_features_savepath
    
    def get_multiclass_clf_features_savepath(self) -> str:
        return self.__multiclass_clf_dataset_with_features_savepath
    
    def get_multiclass_clf_class_labels_path(self) -> str:
        return self.__multiclass_clf_class_labels_path
    
    def get_binary_clf_features_path(self) -> str:
        return self.__dga_binary_clf_features_path
    
    def get_multiclass_clf_features_path(self) -> str:
        return self.__dga_multiclass_clf_features_path
    
    # N-grams
    def get_dga_extract_ngram_dir(self) -> str:
        return self.__dga_extract_ngram_dir
    def get_dga_extract_ngram_full_dir(self) -> str:
        return self.__dga_extract_ngram_full_dir
    def get_non_dga_extract_ngram_dir(self) -> str:
        return self.__non_dga_extract_ngram_dir
    def get_ngrams_dir(self) -> str:
        return self.__ngrams_dir
    
    # Known-TLD
    def get_known_tlds_path(self) -> str:
        return self.__known_tlds_path
    
    # Classification matrix build
    def get_dataset_for_clf_matrix_build(self) -> str:
        return self.__dga_extract_ngram_full_dir
    
    def get_stat_clf_labels_dir(self) -> str:
        return self.__stat_clf_labels_dir
    
    # Proportion picks
    def get_proportion_picked_dataset(self, number) -> str:
        proportion_pick_dir = "./01-Data/02-Preprocessed-data/" \
                              "DGA/0"+ str(number) + "-Proportion-pick/" 
        
        if os.path.exists(proportion_pick_dir):
            return proportion_pick_dir
        else:
            return None
        
    # Median split 
    def get_median_split_dataset_dir(self) -> str:
        median_split_dataset_dir = "./01-Data/02-Preprocessed-data/"\
                                     "DGA/04-Median/"
        return median_split_dataset_dir
    
    # All families - raw-data
    def get_raw_data(self) -> str:
        return "./01-Data/01-Raw-data/dga_archive_full/"