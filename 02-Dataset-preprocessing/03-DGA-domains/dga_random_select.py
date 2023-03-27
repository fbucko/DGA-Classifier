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
Purpose of this script is create dataset from multiple
csv files containing dga domains.
Each file represent one DGA family and
different families have different number of domains.
The differences in the count of the domains is significant.
For training the binary classifier:
1. We build the dataset with proportion pick
2. We build the dataset with random selection
   (without considering minorities in families)
3. We build the dataset with random selection
   including mainly families with small amount of
   samples
   -> set treshold for all selection
   -> set treshold if the proportion is less than zero
"""

import os
import sys
import psutil
import random
import numpy as np
import pandas as pd

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
    
# 1. Determine the total number of records
path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_full/"
filepath = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/DGA-families-count.csv"
edited_path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_datasets/03-Proportion-pick/"
# File contains the number of records for every dga family
dga_stats = pd.read_csv(filepath) 
# dga_stats = dga_stats.drop_duplicates()
# dga_stats.to_csv(filepath, index=False)
print(dga_stats)
total = dga_stats["Num_of_records"].sum()
print(total)

# 2. Determine the total number of records
#    we want to use for training
training_samples = 230_000

# 3. For each file calculate the proportion
#    of records to use for training
dga_stats["Prop"] = (training_samples /  total * dga_stats["Num_of_records"]).round().astype(int)
proportions = dict(zip(dga_stats["DGA_Family"], dga_stats["Prop"]))
print(dga_stats)
print(dga_stats["Prop"].sum())

for idx, dga in enumerate(list(zip(dga_stats["DGA_Family"], dga_stats["Num_of_records"], dga_stats["Prop"]))):
    print(f"{idx}{dga}")

# 4. For each file read a random sample of 
#    records from it using the proportion 
#    calculated in the step 3.
treshold = 3
dfs = []
for idx, (file, prop) in enumerate(proportions.items()):
    # print(prop)
    print(idx)
    if prop > treshold:
        # Proportional pick
        # Generate the random indices
        file_nrows = dga_stats.at[idx,"Num_of_records"]
        random_indices = sorted(random.sample(range(file_nrows), prop))
        
        # Check if the file can fit into memory
        if fit_memory(path + file):
            # Read the whole dataframe and pick random rows
            print(f"{file}: can fit")
            df = pd.read_csv(path + file, header=None)
            tmp_df = pd.DataFrame()
            df = df[df.index.isin(random_indices)]
            tmp_df = df.iloc[:,0]
            tmp_df.to_csv(edited_path + "01-" + file, mode='a', header=False, index=False)
            
        else:
            # 1. Read the file with chunksize
            # 2. Get rows on random indices
            print(f"{file}: can not fit")
            print(f"File_n_rows: {file_nrows}")
            # print("Indices",random_indices)
            
            for chunk in pd.read_csv(path + file, chunksize=100_000, header=None):
                chunk_size = len(chunk)
                # print(chunk_size)
                chunk_start_idx = chunk.index.start
                chunk_end_idx = chunk_start_idx + chunk_size
                # print(chunk_start_idx)
                # print(chunk_end_idx)

                chunk_random_indices = [i for i in random_indices if (i >= chunk_start_idx) and (i < chunk_end_idx)]
                # print(chunk_random_indices)
                if chunk_random_indices:
                    # print(chunk_start_idx)
                    # print(chunk_end_idx)
                    # print(chunk_random_indices)
                    # print(chunk.iloc[chunk_random_indices])
                    chunk_df = pd.DataFrame()
                    chunk = chunk[chunk.index.isin(chunk_random_indices)]
                    chunk_df = chunk.iloc[:,0]
                    chunk_df.to_csv(edited_path + "01-" + file, mode='a', header=False, index=False)
                    
        
    else:
        # Read the whole file
        # Generate the random indices
        file_nrows = dga_stats.at[idx,"Num_of_records"]
        random_indices = sorted(random.sample(range(file_nrows), 3))
        print(file_nrows)
        print(random_indices)
        df = pd.read_csv(path + file, header=None)
        tmp_df = pd.DataFrame()
        df = df[df.index.isin(random_indices)]
        tmp_df = df.iloc[:,0]
        tmp_df.to_csv(edited_path + "01-" + file, mode='a', header=False, index=False)
        
# 5. Concatenate the samples from all files
#    to create the training dataset
#train_data = pd.concat(dfs)