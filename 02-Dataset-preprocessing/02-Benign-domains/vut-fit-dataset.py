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
import pandas as pd

if __name__=="__main__":
    # 1. Read the data from json
    # Data has to be read in chunks, because of the file size
    filepath = "/mnt/c/Work/Bachelors-thesis/Dataset/Non-DGA/VUT-FIT/output.json"
    chunksize = 1000 #Number of rows to read at once
    
    for df in pd.read_json(filepath, lines=True, chunksize=chunksize):
        # print(df.columns)
        # print(df[["domain_name","label"]])
        # 2. Extract the data from column
        new_df = df[["domain_name","label"]].drop_duplicates(subset=["domain_name"])
        # 3. Append them to a new json file
        new_df.to_csv('vut-fit.csv', index=False, mode='a', header=not os.path.exists('vut-fit.csv'))

