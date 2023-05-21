#!/usr/bin/env python3
"""                                                                                                                                                                  3 Filename: regex_classifier.py
Author: Filip Bučko
Date: April 12, 2023
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
Script creates the n-gram database, which contains 2,3,4 grams. 
These ngrams have been proven to be the most efficient in detecting DGA domains.
Input: dataset with domains
Output: file, which contains extracted n-grams
"""

import os
import psutil
import tldextract
import pandas as pd
from dataset_paths import DatasetPaths

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

def extract_ngrams(string:str, n:int) -> list:
    """
    Extracts n-grams from a given text string.

    Args:
        string (str): The input text string.
        n (int): The value of n for n-grams.

    Returns:
        list: A list of n-grams extracted from the input text string.
    """
    n_grams = set( string[i:i+n] for i in range(len(string) - n + 1) )
    return n_grams

def extract_ngrams_from_csv_files(directory:str, n:int, output_file:str):
    """
    Extracts n-grams from strings in multiple CSV files in a directory,
    and stores unique n-grams to a new CSV file.

    Args:
        directory (str): The directory path containing the CSV files.
        n (int): The value of n for n-grams.
        output_file (str): The output CSV file name to store the unique n-grams.

    Returns:
        None
    """
    unique_ngrams = set() # Set to store unique n-grams

    # Iterate over all files in the directory
    # n = 2
    for file in os.listdir(directory):
        if file.endswith(".csv"): # Check if file is a CSV file
            file_path = os.path.join(directory, file)
            if fit_memory(file_path):
                df = pd.read_csv(file_path)
                # # Extract n-grams from each string in the file
                for domain in df.iloc[:, 0]:
                        extracted = tldextract.extract(domain)
                        concatenated_subdomains = (extracted.subdomain + extracted.domain).replace(".","")
                        for n in 2, 3, 4:
                            ngrams = extract_ngrams(concatenated_subdomains, n)
                            unique_ngrams.update(ngrams)
            else:
                for chunk in pd.read_csv(file_path, chunksize=100_000, header=None):
                    for domain in chunk.iloc[:, 0]:
                        extracted = tldextract.extract(domain)
                        concatenated_subdomains = (extracted.subdomain + extracted.domain).replace(".","")
                        for n in 2, 3, 4:
                            ngrams = extract_ngrams(concatenated_subdomains, n)
                            unique_ngrams.update(ngrams)
    # Convert unique n-grams set to a DataFrame and save to CSV
    unique_ngrams_df = pd.DataFrame(list(unique_ngrams), columns=["ngram"])
    unique_ngrams_df.to_csv(output_file, index=False)



if __name__=="__main__":
    dga_directory = DatasetPaths().get_dga_extract_ngram_dir()
    dga_full_directory = DatasetPaths().get_dga_extract_ngram_full_dir()
    non_dga_directory = DatasetPaths().get_non_dga_extract_ngram_dir()
    output_dir = DatasetPaths().get_ngrams_dir()
    
    extract_ngrams_from_csv_files(directory=dga_directory, n=2, output_file= output_dir + "dga-ngram.csv")
    extract_ngrams_from_csv_files(directory=dga_full_directory, n=2, output_file= output_dir + "dga-ngram-full.csv")
    extract_ngrams_from_csv_files(directory=non_dga_directory, n=2, output_file= output_dir + "non-dga-ngram.csv")
    