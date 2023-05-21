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
Purpose of this script is to modify dataset
provided by FETA.
The dataset contains benign domains, which contain
unecessary features for our application.
This script exctracts only domains from json file,
and writes them into csv file
"""

import os
import sys
import pandas as pd

def get_file_paths() -> str:
    """Function returns the directory to the dataset
    of non-dga domains according to the directory
    from which the script where run from

    Returns:
        str: non-dga directory
    """
    cwd = os.getcwd()
    dirname = os.path.basename(cwd)
    if dirname == "01-Benign-domains":
        return "../../01-Data/01-Raw-data/"
    elif dirname == "DGA-Classifier":
        return "./01-Data/01-Raw-data/"
    else:
        return None
    
def get_save_path():
    cwd = os.getcwd()
    dirname = os.path.basename(cwd)
    if dirname == "01-Benign-domains":
        return "../../01-Data/02-Preprocessed-data/Non-DGA/"
    elif dirname == "DGA-Classifier":
        return "./01-Data/02-Preprocessed-data/Non-DGA/"
    else:
        return None
    
def main():
    filename = "cisco-umbrella-nes-fit-verified.json"
    dataset_dir = get_file_paths()
    output_dir = get_save_path()
    if not dataset_dir:
        print("Run the script from the project root directory:",file=sys.stderr)
        exit(1)
    if not output_dir:
        print("Run the script from the project root directory:",file=sys.stderr)
        exit(1)
        
    file = dataset_dir + filename
    chunksize = 1000
    for df in pd.read_json(file, lines=True, chunksize=chunksize):
        print(df.columns)
        print(df[["domain_name","label"]])
        # 2. Extract the data from column
        new_df = df[["domain_name","label"]].drop_duplicates(subset=["domain_name"])
        # 3. Append them to a new json file
        new_df.to_csv(output_dir + 'cisco-umbrella-nes-fit-verified.csv',
                      index=False,
                      mode='a',
                      header=not os.path.exists(output_dir + 'cisco-umbrella-nes-fit-verified.csv'))
    
if __name__=="__main__":
    main()

