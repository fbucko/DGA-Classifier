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
The script creates 2 datasets, which are split by median.
These datasets will be used for binary classifier as a preliminary step
before the multiclass classification itself to refine the multiclass classification.
All informations about the data will be read from one summary csv file

Calculate the median and divide the dataset into 2 groups:
1. Densly represented DGA families
2. Poorly represented DGA families
"""
import os
import sys
import random 
import psutil
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get
sys.path.append("./02-Data-preprocessing/03-Feature-extraction")
from dataset_paths import DatasetPaths

def read_dga_overview() -> pd.DataFrame:
    dga_dir = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/"
    filename = "DGA-families-count.csv"
    file = dga_dir + filename
    
    df = pd.read_csv(file)
    return df

def median_split(df:pd.DataFrame, offset:int = 0) -> tuple:
    """Function splits the dataframe, containing
    overview of DGA families, with a number of samples
    for each family. These dataframes will be used 
    to determine which family will belong to one of the 2 groups

    Args:
        df (pd.DataFrame): Dataframe containing family name
                           and number of samples 
        offset (int): Offset to shift the split of the data
    """
    num_of_samples = df["Num_of_records"]
    median = num_of_samples.median()
    
    families_below_median_df = df[df["Num_of_records"] <= median + offset]
    families_above_median_df = df[df["Num_of_records"] > median + offset]
    
    families_below_median = [family for family in families_below_median_df["DGA_Family"]]
    families_above_median = [family for family in families_above_median_df["DGA_Family"]]
    
    return (families_below_median, families_above_median)

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

def create_dataset_from_families_below_median(families_below_median: list, 
                                              dataset_dir:str,
                                              output_file:str):
    """
    Function creates the new csv file by extracting the first column from csv files,
    where each csv file represents the DGA family.
    
    Args:
        families_below_median (list): list of families, which has
        the number of samples below median
        dataset_dir (str): Directory where are stored family csv files
        output_file (str): Filename of the created dataset
    """
    if os.path.exists(output_file):
        print(f"The file {output_file} already exists.")
        return
    
    dfs = []
    for family_csv in families_below_median:
        df = pd.read_csv(dataset_dir + family_csv, header=None)
        # Extract the first column
        dfs.append(df.iloc[:, 0])

    concat_df = pd.concat(dfs, ignore_index=True)
    concat_df.drop_duplicates(inplace=True)
    
    # Save the combined data into a single CSV file
    concat_df.to_csv(output_file, header=False, index=False)


def generate_random_indices(file:str, proportion: int, dga_stats_df:pd.DataFrame) -> list:
    """Generates random row indices of the rows within the given file.
    Later the rows will be picked according to generated indices. 
    dga_stats_df is used for getting the total number of rows of the file,
    to set a upper limit while generating the random indices.

    Args:
        file (str): File, from which we are going to pick the random indices.
        proportion (int): Indicates how many random indices should be generated.
        dga_stats_df (pd.DataFrame): DGA Families overview dataframe.

    Returns:
        list: List of random row indices
    """
    file_rows_count = dga_stats_df.loc[dga_stats_df["DGA_Family"] == file, "Num_of_records"].values[0]
    random_row_indices = sorted(random.sample(range(file_rows_count), proportion))
    return random_row_indices
    
def create_dataset_from_families_above_median(dga_stats_df:pd.DataFrame,
                                              proportions:dict,
                                              dataset_dir:str,
                                              output_file:str):
    """The difference between 'create_dataset_from_families_below_median'
    function, is that we have pick the dga families randomly and proportionaly
    to have every family represented.

    Args:
        families_above_median (list): list of families, which has
        the number of samples above median
        proportions (dict): Dictionary, where key is the name of the file
                            and value is the number of samples
                            which should be picked.
        dataset_dir (str): Directory where are stored family csv files
        output_file (str): Filename of the created dataset
        proportions
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists.")
        return
    
    for file, proportion in proportions.items():
        random_row_indices = generate_random_indices(file, proportion, dga_stats_df)
        # 1. Check if the files fits into the memory
         # Check if the file can fit into memory
        if fit_memory(dataset_dir + file):
            # Read the whole dataframe and pick random rows
            print(f"{file}: can fit")
            df = pd.read_csv(dataset_dir + file, header=None)
            tmp_df = pd.DataFrame()
            df = df[df.index.isin(random_row_indices)]
            tmp_df = df.iloc[:,0]
            tmp_df.to_csv(output_file, mode='a', header=False, index=False)
            
        else:
            print(f"{file}: can not fit")
            for chunk in pd.read_csv(dataset_dir + file, chunksize=100_000, header=None):
                chunk_size = len(chunk)
                chunk_start_idx = chunk.index.start
                chunk_end_idx = chunk_start_idx + chunk_size

                chunk_random_indices = [i for i in random_row_indices if (i >= chunk_start_idx) and (i < chunk_end_idx)]
                if chunk_random_indices:
                    chunk_df = pd.DataFrame()
                    chunk = chunk[chunk.index.isin(chunk_random_indices)]
                    chunk_df = chunk.iloc[:,0]
                    chunk_df.to_csv(output_file, mode='a', header=False, index=False)
    
def count_class_proportions(dga_stats:pd.DataFrame, families_above_median: list, dataset_size: int) -> dict:
    """The function is necessary when creating a dataset from classes that
    exceed the median. The goal is to select a randomly determined number of dga domains,
    so that the number from each class is represented in proportion

    Args:
        df (pd.DataFrame): Dataframe containing Family names and number of samples for each family
        families_above_median (list): Family names which exceeds the median
        dataset_size (int) : The number of elements from which the resulting dataset should be composed
    Returns:
        dict: Dictionary, where key is the name of the file and value is the number of samples
              which should be picked. 
    """
    above_median_df = dga_stats[dga_stats["DGA_Family"].isin(families_above_median)].copy()
    total = above_median_df["Num_of_records"].sum()
    above_median_df["Prop"] = (dataset_size /  total * dga_stats["Num_of_records"]).round().astype(int)
    proportions = dict(zip(above_median_df["DGA_Family"], above_median_df["Prop"]))
    
    return proportions
    
def create_binary_preclassificaion_datasets(dga_stats_df:pd.DataFrame,
                                            families_below_median:list,
                                            families_above_median:list):
    """Function creates the datasets for binary pre-classification
    to reduce the class distribution imbalances.
    That's why the classification of currently 93 classes
    will be split into 2 parts and classifying the 46/47 classes
    in each part

    Args:
        dga_stats (pd.DataFrame): Statistical info about all csv files,
                                  where every csv file contains DGA Family samples.
        families_below_median (list): List of families whose number of samples does
                                      not exceed the median
        families_above_median (list): List of families whose number of samples does
                                      exceed the median
    """
     # 1. Find out which families are going to be split
    # and create the labels of 2 groups
    
    dga_families_samples_dir = DatasetPaths().get_raw_data()
    output_dataset_dir = DatasetPaths().get_median_split_dataset_dir()
    below_output_dataset_file = output_dataset_dir + "01-classes-below-median.csv"
    above_output_dataset_file = output_dataset_dir + "01-classes-above-median.csv"
    
    # 2. Create the 2 datasets:
    #   a) Poorly represented DGA Families (Pick all classes)
    #   b) Densly represented DGA Families (Random proportion pick -> check if it fits the memory)
    create_dataset_from_families_below_median(families_below_median,
                                              dga_families_samples_dir,
                                              below_output_dataset_file
                                              )
    proportions = count_class_proportions(dga_stats_df, families_above_median, dataset_size=150_000)
    create_dataset_from_families_above_median(dga_stats_df,
                                              proportions,
                                              dga_families_samples_dir,
                                              above_output_dataset_file,
                                              )
    
def create_dataset_for_poorly_represented_dga(dga_stats:pd.DataFrame,
                                              families_below_median:list,
                                              dataset_dir:str,
                                              output_file:str = None):
    """Creates the csv file from multiple csv files, where every csv file 
    contains samples of certain DGA family. While creating the one dataset,
    it's necessary to label families for later classification. 

    Args:
        dga_stats (pd.DataFrame): DGA Families overview 
                            (DGA Family names, Number of samples for every family)
        families_below_median (list): Families whose number of samples does not exceed the median
        dga_families_samples_dir (str): The directory, from which we will be reading
                                           the csv files
        below_output_dataset_file (str): Resulting dataset in csv file format
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists.")
        return
    
    dfs = []
    for family_csv in families_below_median:
        df = pd.read_csv(dataset_dir + family_csv, header=None)
        # Add class labels
        df["DGA_Family"] = family_csv[:-4] # -4 to remove '.csv' suffix
        # Extract the first column
        first_column = df.iloc[:, 0]
        dga_family_column = df["DGA_Family"]
        df = pd.DataFrame({0: first_column, 1: dga_family_column})
        dfs.append(df)

    concat_df = pd.concat(dfs, ignore_index=True)
    concat_df.drop_duplicates(subset=concat_df.columns[0], inplace=True)
    
    # Save the combined data into a single CSV file
    if output_file and not os.path.exists(output_file):
        concat_df.to_csv(output_file, index=False, header=False)
    
def create_dataset_for_densly_represented_dga(dga_stats_df:pd.DataFrame,
                                              families_above_median:list,
                                              dataset_dir:str,
                                              output_file:str = None):
    """Creates the csv file from multiple csv files, where every csv file 
    contains samples of certain DGA family. While creating the one dataset,
    it's necessary to label families for later classification. 

    Args:
        dga_stats (pd.DataFrame): DGA Families overview 
                            (DGA Family names, Number of samples for every family)
        families_below_median (list): Families whose number of samples does exceed the median
        dga_families_samples_dir (str): The directory, from which we will be reading
                                           the csv files
        below_output_dataset_file (str): Resulting dataset in csv file format
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists.")
        return
    
    proportions = count_class_proportions(dga_stats_df, families_above_median, dataset_size=1_000_000)
    for file, proportion in proportions.items():
        random_row_indices = generate_random_indices(file, proportion, dga_stats_df)
        # 1. Check if the files fits into the memory
         # Check if the file can fit into memory
        if fit_memory(dataset_dir + file):
            # Read the whole dataframe and pick random rows
            print(f"{file}: can fit")
            df = pd.read_csv(dataset_dir + file, header=None)
            df = df[df.index.isin(random_row_indices)]
            df["DGA_Family"] = file[:-4] # -4 to remove '.csv' suffix
            tmp_df = pd.DataFrame({0:df.iloc[:,0], 1:df["DGA_Family"]})
            if output_file:
                tmp_df.to_csv(output_file, mode='a', header=False, index=False)
            
        else:
            for chunk in pd.read_csv(dataset_dir + file, chunksize=100_000, header=None):
                chunk_size = len(chunk)
                chunk_start_idx = chunk.index.start
                chunk_end_idx = chunk_start_idx + chunk_size

                chunk_random_indices = [i for i in random_row_indices if (i >= chunk_start_idx) and (i < chunk_end_idx)]
                if chunk_random_indices:
                    chunk = chunk[chunk.index.isin(chunk_random_indices)]
                    chunk["DGA_Family"] = file[:-4] # -4 to remove '.csv' suffix
                    chunk_df = pd.DataFrame({0:chunk.iloc[:,0], 1:chunk["DGA_Family"]})
                    if output_file:
                        chunk_df.to_csv(output_file, mode='a', header=False, index=False)

def create_datasets_for_multiclass_clf(dga_stats:pd.DataFrame,
                                       families_below_median:list,
                                       families_above_median:list):
    """Function that creates 2 datasets for multiclass classification.
    First dataset consist of 47 classes of poorly represented classes.
    The second dataset contains the dataset of densly represented classes. 

    Args:
        dga_stats (pd.DataFrame): DGA stats overview
        families_below_median (list): List of families whose number of samples does
                                      not exceed the median
        families_above_median (list): List of families whose number of samples does
                                      exceed the median
    """
    dga_families_samples_dir = DatasetPaths().get_raw_data()
    output_dataset_dir = DatasetPaths().get_median_split_dataset_dir()
    below_output_dataset_file = output_dataset_dir + "02-poorly-represented-dga.csv"
    above_output_dataset_file = output_dataset_dir + "02-densly-represented-dga.csv"

    create_dataset_for_poorly_represented_dga(dga_stats,
                                              families_below_median,
                                              dga_families_samples_dir,
                                              below_output_dataset_file
                                              )
    create_dataset_for_densly_represented_dga(dga_stats,
                                              families_above_median,
                                              dga_families_samples_dir,
                                              above_output_dataset_file
                                              )
    
    
def main():
    df = read_dga_overview()
    families_below_median, families_above_median = median_split(df, offset=10_000)
    create_binary_preclassificaion_datasets(df,
                                            families_below_median,
                                            families_above_median
                                            )
    create_datasets_for_multiclass_clf(df,
                                       families_below_median,
                                       families_above_median)
if __name__=="__main__":
    main()
