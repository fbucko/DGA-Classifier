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
Purpose of this script is to compute specified features
for domains in datasets. Features will be used for DGA
binary classification.
Selected features:
1. Total domain length
2. TLD length
3. SLD length
4. The length of the longest consonant sequence in SLD
5. Number of digits in SLD
6. Number of unique characters in the domain string (TLD + SLD)
7. Digit ratio -> number of digits divided by string length
8. Consonant ratio -> number of consonants divided by string length
9. Non-alfanumeric ratio
10. Hexadecimal ratio
11. Flag of beginning number
12. Flag of malicious domain -> : “study”, “party”, “click”, “top”, “gdn”, “gq”, “asia”,
“cricket”, “biz”, “cf”.
13. Normalized entropy of the string
14. N - gram features
    -> extract ngrams from datasets -> 2,3,4 gram -> build dictionaries -> save them to separate file
    -> Create 2 files for bening N-grams and for DGA N-grams
    -> Calculate the ratio for 2,3,4 gram by counting the number of occurence in Benign and DGA hashtables
       and dividing the amount of n-grams of the string
    -> calculate the average value for 2,3,4 ngrams for Benign and DGA classes
15. DGA similarity -> abs(DGA n-gram average - Non-dga average)
16. Dictionary matching ratio -> length of matched words / total domain length
import nltk
nltk.download('words')

english_vocab = set(w.lower() for w in nltk.corpus.words.words())

def is_english_word(word):
    return word.lower() in english_vocab

# Example usage:
print(is_english_word('hello')) # Output: True
print(is_english_word('hola')) # Output: False

--------------
import requests

def is_english_word(word):
    url = f'https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key=<your_api_key_here>'
    response = requests.get(url)
    if response.ok:
        results = response.json()
        return len(results) > 0
    else:
        response.raise_for_status()

# Example usage:
print(is_english_word('hello')) # Output: True
print(is_english_word('hola')) # Output: False
17. Contains www
"""
import math
import pickle
import tldextract
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score


def domain_len(domain: str) -> int:
    """Returns length of the domain

    Args:
        domain (str): domain name

    Returns:
        int: length of the domain
    """
    return len(domain)

def longest_consonant_seq(domain: str) -> int:
    """Function returns longest consonant sequence

    Args:
        domain (str): domain name

    Returns:
        int: length of the longest consonant sequence
    """
    consonants = "bcdfghjklmnpqrstvwxyz"
    current_len = 0
    max_len = 0
    domain = domain.lower()
    for char in domain:
        if char in consonants:
            current_len += 1
        else:
            current_len = 0
        if current_len > max_len:
            max_len = current_len
    return max_len

def count_digits(sld_domain:str) -> int:
    """Returns the count of digits in the second
    leve domain

    Args:
        sld_domain (str): second level domain

    Returns:
        int: the number of digits in the SLD
    """
    count = 0
    for char in sld_domain:
        if char.isdigit():
            count += 1
    return count

def unique_character_count(domain: str) -> int:
    """Fuction returns the number of unique characters
    in the SLD and TLD domain

    Args:
        domain (str): SLD + TLD domain -> concatenated

    Returns:
        int: Number of domain count
    """
    # Remove dots if the TLD is composed from country code
    domain = domain.replace(".", "")
    return len(set(domain))

def get_consonant_ratio(domain: str) -> float:
    """Function returns the consonant ratio
    which represents the total amount of consonants
    divided by the string length

    Args:
        domain (str): domain 

    Returns:
        float: consonant ratio
    """
    domain = domain.lower()
    consonants = set("bcdfghjklmnpqrstvwxyz")
    consonant_count = sum( 1 for char in domain if char in consonants )
    domain_len = len(domain)
    return consonant_count / domain_len if consonant_count > 0 else 0.0

def get_non_alfa_numeric_ratio(domain: str) -> float:
    """Function returns the ratio of non-alfanumeric characters
    which represents the total amount of non-alfanumeric chars
    divided by the string length

    Args:
        domain (str): domain

    Returns:
        float: the ratio of non-alfanumeric characters
    """
    non_alphanumeric = sum(not char.isalnum() for char in domain)
    return non_alphanumeric / len(domain)

def get_hex_ratio(domain: str) -> float:
    """Function computes hexadecimal ratio

    Args:
        domain (str): The length of the domain

    Returns:
        float: hexadecimal ratio
    """
    hex_chars = set('0123456789ABCDEFabcdef')
    hex_count = sum(char in hex_chars for char in domain)
    return hex_count / len(domain)

def flag_of_first_digit(domain: str) -> int:
    """Function returns whether the string starts with the digits.

    Args:
        domain (str): domain

    Returns:
        int: 1 if the string starts with the digit.
             0 if the string does not start with the digit
        
    """
    if len(domain) == 0:
        return 0
    return int(domain[0].isdigit())

def get_normalized_entropy(domain: str) -> float:
    """Function returns the normalized entropy of the
    string. The function first computes the frequency
    of each character in the string using
    the collections.Counter function.
    It then uses the formula for entropy to compute
    the entropy of the string, and normalizes it by
    dividing it by the maximum possible entropy
    (which is the logarithm of the minimum of the length
    of the string and the number of distinct characters
    in the string).

    Args:
        domain (str): domain string

    Returns:
        float: normalized entropy
    """
    domain_len = len(domain)
    freqs = {}
    for char in domain:
        if char in freqs:
            freqs[char] += 1
        else:
            freqs[char] = 1
    
    entropy = 0.0
    for f in freqs.values():
        p = float(f) / domain_len
        entropy -= p * math.log(p, 2)
    return entropy / domain_len
    # domain_len = len(domain)
    # if domain_len == 0:
    #     return 0.0
    # freqs = Counter(domain)
    # print(freqs)
    # entropy = -sum(count/domain_len * math.log(count/domain_len, 2) for count in freqs.values())
    # max_entropy = math.log(min(domain_len, len(freqs)), 2)
    # if max_entropy != 0:
    #     return entropy / max_entropy
    # else:
    #     return entropy

def extract_features(domain: str):
    """
    Function extracts specified features and writes
    them to the pandas dataframe

    Args:
        domain (str): fully specified domain name
    """
    # - Extract subdomains from domains
    ext = tldextract.extract(domain)
    ext_domain = ext.domain
    ext_suffix = ext.suffix
    
    if not ext_domain:
        print(f"{domain}")
        #Parse suffix
        #How many words are delimited
        domain_count = ext_suffix.split(".")
        print(domain_count)
        if len(domain_count) > 1:
            ext_domain = domain_count[-2]
            ext_suffix = domain_count[-1]
            
    # 1. Domain length
    domain_length = domain_len(ext_domain)
    # 2. TLD length
    tld_len = domain_len(ext_suffix)
    # 3. SLD length
    sld_len = domain_len(ext_domain)
    # 4. The length of the longest consonant sequence
    max_consonant_len = longest_consonant_seq(ext_domain)
    # 5. The number of digits in the domain
    sld_digits_len = count_digits(ext_domain)
    # 6. The number of unique characters in the domain (SLD + TLD)
    unique_chars = unique_character_count(ext_domain + ext_suffix)
    # 7. Digit ratio
    digit_ratio = sld_digits_len / domain_length
    # 8. Consonant ratio
    consonant_ratio = get_consonant_ratio(ext_domain)
    # 9. Non-alfanumeric ratio
    non_alfa_ratio = get_non_alfa_numeric_ratio(ext_domain)
    # 10. Hexadecimal ratio
    hex_ratio = get_hex_ratio(ext_domain)
    # 11. Flag of beginning digit
    first_digit_flag = flag_of_first_digit(ext_domain)
    # 12. Flag of malicous domain -> Flag of well-known TLD NOPE for now
    # 13. Normalized entropy
    norm_entropy = get_normalized_entropy(ext_domain)
    
    return (
        domain_length,
        tld_len,
        sld_len,
        max_consonant_len,
        sld_digits_len,
        unique_chars,
        digit_ratio,
        consonant_ratio,
        non_alfa_ratio,
        hex_ratio,
        first_digit_flag,
        norm_entropy)
    """
    1. Total domain length
2. TLD length
3. SLD length
4. The length of the longest consonant sequence in SLD
5. Number of digits in SLD
6. Number of unique characters in the domain string (TLD + SLD)
7. Digit ratio -> number of digits divided by string length
8. Consonant ratio -> number of consonants divided by string length
9. Non-alfanumeric ratio
10. Hexadecimal ratio
11. Flag of beginning number
12. Flag of malicious domain -> : “study”, “party”, “click”, “top”, “gdn”, “gq”, “asia”,
“cricket”, “biz”, “cf”.
13. Normalized entropy of the string
14. N - gram features
    -> extract ngrams from datasets -> 2,3,4 gram -> build dictionaries -> save them to separate file
    -> Create 2 files for bening N-grams and for DGA N-grams
    -> Calculate the ratio for 2,3,4 gram by counting the number of occurence in Benign and DGA hashtables
       and dividing the amount of n-grams of the string
    -> calculate the average value for 2,3,4 ngrams for Benign and DGA classes
15. DGA similarity -> abs(DGA n-gram average - Non-dga average)
16. Dictionary matching ratio -> length of matched words / total domain length
    """
if __name__=="__main__":
    path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_datasets/01-Proportion-pick/"
    path2 = "/mnt/c/Work/Bachelors-thesis/Dataset/Non-DGA/VUT-FIT/"
    file = "00-all.csv"
    file2 = "vut-fit.csv"
    df = pd.read_csv(path + file, names=["domain"])
    df2 = pd.read_csv(path2 + file2, names=["domain"])
    
    # - Extract subdomains from domains
    df[["domain_len",
        "tld_len",
        "sld_len",
        "max_consonant_len",
        "sld_digits_len",
        "unique_chars",
        "digit_ratio",
        "consonant_ratio",
        "non_alfa_ratio",
        "hex_ratio",
        "first_digit_flag",
        "norm_entropy"]] = df["domain"].apply(extract_features).apply(pd.Series)
    
    df2[["domain_len",
        "tld_len",
        "sld_len",
        "max_consonant_len",
        "sld_digits_len",
        "unique_chars",
        "digit_ratio",
        "consonant_ratio",
        "non_alfa_ratio",
        "hex_ratio",
        "first_digit_flag",
        "norm_entropy"]] = df2["domain"].apply(extract_features).apply(pd.Series)
    
    df["dga"] = 1
    df2["dga"] = 0
    
    final_df = pd.concat([df,df2], ignore_index=True)
    # print(final_df)
    
    # CREATE XGBOOST MODEL
    # Split the data
    X = final_df[["domain_len",
        "tld_len",
        "sld_len",
        "max_consonant_len",
        "sld_digits_len",
        "unique_chars",
        "digit_ratio",
        "consonant_ratio",
        "non_alfa_ratio",
        "hex_ratio",
        "first_digit_flag",
        "norm_entropy"]]
    y = final_df["dga"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the XGBoost model
    params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print('ROC AUC:', roc_auc)
    print('Scores:', scores.mean())

    #Export model to file
    # with open('model.pkl', 'wb') as f:
        # pickle.dump(model, f)