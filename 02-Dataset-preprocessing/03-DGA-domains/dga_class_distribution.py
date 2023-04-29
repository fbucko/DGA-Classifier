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
This script is used to analyze the representation of each class.
Calculate the median and divide the dataset into 2 groups:
1. Densly represented DGA families
2. Poorly represented DGA families
"""

import pandas as pd
import matplotlib.pyplot as plt

def median_split(df:pd.DataFrame):
    """
    Prints the data divided by median

    Args:
        df (pd.DataFrame): dataframe containing class distribution
    """
    # 2. Extract the columns for class name and number of samples
    # dga_families = df["DGA_Family"].str.replace("_dga.csv","", regex=False)
    num_samples = df["Num_of_records"]
    median = num_samples.median()
    print(median)
    families_above_median = df[df["Num_of_records"] > median + 10_000]
    families_below_median = df[df["Num_of_records"] <= median + 10_000]
    print(families_below_median["Num_of_records"].sum())
    print(families_above_median["Num_of_records"].sum())
    
    # Print the classes above and below the median
    print('Classes Above Median:')
    print(families_above_median)
    print(families_above_median["Num_of_records"].sum())
    
    print('\nClasses Below Median:')
    print(families_below_median)
    print(families_below_median["Num_of_records"].sum())
    
def quartile_split(df: pd.DataFrame):
    """
    Prints the data divided by quartiles

    Args:
        df (pd.DataFrame): dataframe containing class distribution
    """
    num_samples = df["Num_of_records"]
    # Quantiles
    q1 = num_samples.quantile(0.25)
    q2 = num_samples.quantile(0.50)
    q3 = num_samples.quantile(0.75)
    
    # Divide dataframe into quartiles
    df_q1 = df[df["Num_of_records"] <= q1]  # Quartile 1 (Q1)
    df_q2 = df[(df["Num_of_records"] > q1) & (df["Num_of_records"] <= q2)]  # Quartile 2 (Q2)
    df_q3 = df[(df["Num_of_records"] > q2) & (df["Num_of_records"] <= q3)]  # Quartile 3 (Q3)
    df_q4 = df[df["Num_of_records"] > q3]  # Quartile 4 (Q4)
    
    # Print the resulting dataframes
    print("Quartile 1 (Q1):")
    print(df_q1)
    print(df_q1["Num_of_records"].sum())
    
    print("\nQuartile 2 (Q2):")
    print(df_q2)
    print(df_q2["Num_of_records"].sum())

    print("\nQuartile 3 (Q3):")
    print(df_q3)
    print(df_q3["Num_of_records"].sum())

    print("\nQuartile 4 (Q4):")
    print(df_q4)
    print(df_q4["Num_of_records"].sum())
    
    
def visualize_class_distribution(dga_families:pd.Series, num_samples:pd.Series):
    """
    Function visualizes the distribution
    of DGA classes
    """

    
    # 3. Create a bar chart for data distribution in each class 
    plt.figure(figsize=(12,8))
    plt.barh(dga_families, num_samples, height=2)
    # plt.xticks(rotation=90)
    plt.xlabel('Family Name')
    plt.ylabel('Number of Samples')
    plt.title('Data distribution by DGA Family')
    plt.show()
    
def main():
    dga_dir = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/"
    filename = "DGA-families-count.csv"
    file = dga_dir + filename
    
    # 1. Read the csv file into pandas dataframe 
    df = pd.read_csv(file)
    
    median_split(df)
    # quartile_split(df)
    
    
    
    
    
    visualize = False
    # if visualize:
    #     # Create subplots with two side-by-side plots
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    #     # Plot classes above the median
    #     ax1.bar(families_above_median['DGA_Family'].str.replace("_dga.csv", "", regex=False), families_above_median['Num_of_records'])
    #     ax1.set_xlabel('Class')
    #     ax1.set_ylabel('Number of Samples')
    #     ax1.set_title('Classes Above Median')
    #     # Plot classes below the median
    #     ax2.bar(families_below_median['DGA_Family'].str.replace("_dga.csv","", regex=False), families_below_median['Num_of_records'])
    #     ax2.set_xlabel('Class')
    #     ax2.set_ylabel('Number of Samples')
    #     ax2.set_title('Classes Below Median')

    #     # Adjust spacing between subplots
    #     plt.subplots_adjust(wspace=0.4)

    #     # Rotate x-axis labels in the first subplot
    #     plt.setp(ax1.get_xticklabels(), rotation=90, ha='right')
    #     plt.setp(ax2.get_xticklabels(), rotation=90, ha='right')


        # Show the plot
        # plt.show()
        
        # print(f"75th percentile: {num_samples.quantile(0.75)}")
        # visualize_class_distribution(dga_families, num_samples)
    
if __name__=="__main__":
    main()