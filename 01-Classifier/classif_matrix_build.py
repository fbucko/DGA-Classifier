"""
Filename: classif_matrix_build.py
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
This scripts creates matrix of mean values of selected features
for statistical classification for the dga detection. 
The result of the script is to calculate the 
classification matrix according to dga dataset.
The classification matrix consists of mean values
of the extracted features from the DGA dataset
"""

import os
import sys
import psutil
import numpy as np
import pandas as pd

sys.path.append("/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/02-Dataset-preprocessing/04-Feature-extraction")
from n_grams import N_grams
from known_tld import KnownTLD
from extract_features import extract_features, extract_features_2


def fit_memory(filepath:str) -> bool:
    """
    Function checks whether the file can be loaded into memory. 
    If there is enough room, the function returns True.
    If the file cannot be loaded into memory, returns False

    Args:
        filepath (str): Filepath to the analized file

    Returns:
        bool: True if the file can be read into memory.
              False if the file can not be read into memory
    """
    file_size = os.stat(filepath).st_size
    available_memory = psutil.virtual_memory().available
    
    # Get dataframe sample size
    sample_size = 100
    df_sample = pd.read_csv(filepath, nrows=sample_size)
    # Calculate the memory usage
    memory_usage = df_sample.memory_usage(deep=True).sum()
    # Calculate the expected memory usage
    expected_memory_usage = memory_usage * (file_size / sample_size)
    
    #Return the result
    if expected_memory_usage < available_memory:
        return True
    else:
        return False

def get_classsif_matrix():
    """Function creates classification matrix
    which contains mean values of features for all families
    """
    path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_full/"
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
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            
            # Check if the file can fit into memory
            if fit_memory(path + file):
                print(f"{file}: can fit")
                df = pd.read_csv(file_path,
                                 header=None,
                                 usecols=[0],
                                 names=["domain"])
                print(df)
                df[feature_columns] = df["domain"].apply(extract_features).apply(pd.Series)
                print(df.mean())
                print(df)
            else:
                # 1. Read the file with chunksize
                # 2. Get rows on random indices
                print(f"{file}: can not fit")
                sums = None
                counts = None
                for chunk in pd.read_csv(file_path,
                                         chunksize=100_000,
                                         header=None,
                                         usecols=[0]):
                    chunk[feature_columns] = chunk.iloc[:,0].apply(extract_features).apply(pd.Series)
                    # Compute the sum and count of each feature in the current chunk
                    chunk_sums = chunk.sum()
                    chunk_counts = chunk.count()

                    # Initialize the sums and counts variables if they haven't been initialized yet
                    if sums is None:
                        sums = chunk_sums
                        counts = chunk_counts
                    # Otherwise, add the sums and counts for the current chunk to the existing values
                    else:
                        sums += chunk_sums
                        counts += chunk_counts

def init_known_tld():
    known_subdomain_path = "/mnt/c/Work/Bachelors-thesis/Dataset/Non-DGA/public_suffix_list.dat.txt"
    tlds = KnownTLD(known_subdomain_path)
    return tlds.get_tlds()
    
def init_n_grams():
    ngram_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/03-Models/ngrams/"
    dga_ngram_csv = ngram_dir + "dga-ngram.csv"
    nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
    return N_grams(dga_ngram_csv=dga_ngram_csv, nondga_ngram_csv=nondga_ngram_csv)
    
def get_classsif_matrix2():
    """Function creates classification matrix
    which contains mean values of features for all families
    """
    mean_df = pd.DataFrame()
    path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_full/"
    
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
    
    known_tlds = init_known_tld()
    ngrams = init_n_grams()
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)    
            # 1. Read file into dataframe
            df = pd.read_csv(file_path,
                                 header=None,
                                 usecols=[0],
                                 names=["domain"],
                                 nrows=260_000)
            # 2. Extract features
            df[feature_columns] = df["domain"].apply(lambda x: extract_features_2(x, known_tlds, ngrams)).apply(pd.Series)
            # 3. Compute mean values
            feature_means = df.mean(numeric_only=True)
            mean_df[os.path.splitext(file)[0]] = feature_means
            # print(mean_df)
            # new_row = pd.DataFrame(feature_means).T
            # new_df = pd.DataFrame([new_row], index=[os.path.splitext(file)[0]])
            # print(new_row)

            # print(new_df)
            # print(mean_df)

            # mean_df = pd.concat([mean_df, new_row], axis=0)
            # mean_df[os.path.splitext(file)[0]] = new_row
            # mean_df = mean_df.append(new_df)
    mean_df = mean_df.T
    print(mean_df)
    # mean_df.to_csv("classif_matrix1")
    mean_df.to_csv("classif_matrix", index_label="row_name")
    
    # mean_df.to_json("classif_matrix3")
    # mean_df.to_json("classif_matrix4", orient='index')
    
if __name__=="__main__":
    get_classsif_matrix2()
    pass