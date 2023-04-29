import os
import argparse
import pandas as pd

def parse_script_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Script for removing duplicates from csv files")
    # Add an argument for the filename
    parser.add_argument("-f","--file", type=str, help="Name of the csv file to process")
    parser.add_argument("-o","--out-file", dest="outfile", type=str, help="Name of the modified csv file to be saved")
    # Parse the command-line arguments
    args = parser.parse_args()
    return args

def remove_duplicates(file_csv:str, output_csv:str=None) -> pd.DataFrame:
    """Removes duplicates from given csv file, returns the dataframe
    without duplicates.

    Args:
        file_csv (str): csv file from which are going to be duplicates removed
        output_csv (str, optional): If the parameter is given, the modified
        file without duplicates is saved on given path. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame without duplicates
    """
    df = pd.read_csv(file_csv, header=None)
    df.drop_duplicates(subset=df.columns[0], inplace=True)
    if output_csv:
        df.to_csv(output_csv, index=False, header=False)
    return df
    
def main():
    args = parse_script_arguments()
    file = args.file
    out_file = args.outfile
    remove_duplicates(file_csv=file, output_csv=out_file)
    
if __name__=="__main__":
    main()