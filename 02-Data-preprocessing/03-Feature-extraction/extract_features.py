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
1. Total domain length -> done
2. TLD length -> done
3. SLD length -> done
4. The length of the longest consonant sequence in SLD -> done
5. Number of digits in SLD -> done
6. Number of unique characters in the domain string (TLD + SLD) -> done
7. Digit ratio -> number of digits divided by string length -> done
8. Consonant ratio -> number of consonants divided by string length -> done
9. Non-alfanumeric ratio -> done
10. Hexadecimal ratio -> done
11. Flag of beginning number -> done
12. Flag of well known domain -> : “study”, “party”, “click”, “top”, “gdn”, “gq”, “asia”,
“cricket”, “biz”, “cf”. -> done
13. Normalized entropy of the string -> done
14. N - gram features
    -> extract ngrams from datasets -> 2,3,4 gram -> build dictionaries -> save them to separate file
    -> Create 2 files for bening N-grams and for DGA N-grams
    -> Calculate the ratio for 2,3,4 gram by counting the number of occurence in Benign and DGA hashtables
       and dividing the amount of n-grams of the string
    -> calculate the average value for 2,3,4 ngrams for Benign and DGA classes
15. DGA similarity -> abs(DGA n-gram average - Non-dga average)
16. Dictionary matching ratio -> length of matched words / total domain length -> done
17. Number of subdomains -> done
18. Well-known tld -> done 
19. Contains www -> done
"""
import math
import time
import nltk
import pickle
import tldextract
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import words
from n_grams import N_grams
from known_tld import KnownTLD
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
# Specify custom data path
nltk.data.path.append('./04-Models')
# Download data to custom path
nltk.download('words', download_dir='./04-Models')
english_words = set(words.words())

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

def contains_www(domain: str) -> int:
    """
    Function returns whether the domain contains
    the www subdomain. If the last subdomain is 'www'
    function returns 1, Otherwise function returns 0

    Args:
        domain (str): The whole domain name

    Returns:
        int: 1 if the domain name contains 'www'
             0 if the domain name does not contain 'www'
    """
    subdomains = domain.split(".")
    if subdomains[0] == "www":
        return 1
    return 0

def count_subdomains(domain:str) -> int:
    """
    Function returns the number of subdomains
    in the domain name 

    Args:
        domain (str): The domain name

    Returns:
        int: Number of subdomains
    """
    ext = tldextract.extract(domain)
    if not ext.subdomain:
        return 0

    else:
        subdomains = ext.subdomain.split(".")
        subdomains_count = len(subdomains)
        if "www" in subdomains:
            subdomains_count -= 1
        return subdomains_count

def verify_tld(domain_suffix:str, known_tlds:set) -> int:
    """
    Function checks whether the domain tld is in 
    the public suffix database

    Args:
        domain_suffix (str): Domain tld
        known_tlds (set): The set of the known tlds

    Returns:
        int: 1 if the tld is well-known
             0 if the tld is not well-known
    """
    if domain_suffix in known_tlds:
        return 1
    else:
        return 0

def remove_tld(domain: str) -> str:
    """Function removes tld from
    the domain name

    Args:
        domain (str): Domain name

    Returns:
        str: Domain without TLD
    """
    ext =  tldextract.extract(domain)
    subdomain = ext.subdomain
    sld = ext.domain
    result = subdomain + "." + sld if subdomain else sld
    return result

def extract_subdomains(domain:str) -> list:
    """
    Function returns the list of the subdomains and 
    sld from domain name

    Args:
        domain (str): The domain name

    Returns:
        list: Subdomains not including tld
    """
    ext = tldextract.extract(domain)
    subdomains = ext.subdomain.split('.') if ext.subdomain else []
    if 'www' in subdomains:
        subdomains.remove('www')
    sld = ext.domain
    domain_list = subdomains + [sld]
    return domain_list

def find_longest_word(char_sequence:str) -> list:
    """
    Function find the longest valid English word
    in a given sequence of characters.

    Args:
    - char_sequence (str): Input sequence of characters

    Returns:
    - str: Longest valid English word found in the input sequence
    """
    matched_words = []
    word = ""
    longest_word = None
    # If the empty string is passed
    if not char_sequence:
        return 
    
    # Iterate thorugh string 
    for char in char_sequence:
        word += char
        if word in english_words:
            matched_words.append(word)

    if matched_words:
        longest_word = max(matched_words, key=len)
        if longest_word:
            if (len(char_sequence) - len(longest_word)) > 0:
                find_longest_word(char_sequence.replace(longest_word, ""))
    else:
        find_longest_word(char_sequence[1:])
    return longest_word
    
def find_longest_matched_words(char_sequence: str) -> list:
    """
    Function finds the longest valid English word(s)
    in a given sequence of characters.

    Args:
    - char_sequence (str): Input sequence of characters
    - english_words (set): Set of valid English words

    Returns:
    - list: List of longest matched English words found in the input sequence
    """
    matched_words = []
    word = ""
    longest_matched_words = []
    
    if not char_sequence:
        return longest_matched_words
    
    for char in char_sequence:
        word += char
        if word in english_words and len(word) > 1:
            matched_words.append(word)

    if matched_words:
        longest_word_length = max(len(word) for word in matched_words)
        longest_matched_words = [word for word in matched_words if len(word) == longest_word_length]
        if len(char_sequence) - longest_word_length > 0:
            return longest_matched_words + find_longest_matched_words(char_sequence[longest_word_length:])
        else:
            return longest_matched_words
    else:
        return find_longest_matched_words(char_sequence[1:])
    
def dictionary_matching(domain:str)-> float:
    """A function to find the longest English
    word in a sequence of characters. 
    The function finds all longest words in a sequence
    and returns the following ratio:
    (lenght of matched words) / (length of a sequence)

    Args:
        domain (str): Domain name without TLD

    Returns:
        float: Ratio of the matched words to the sequence length
    """
    matched_words = find_longest_matched_words(domain)
    words_length = 0
    domain_length = len(domain.replace(".",""))
    words_length = sum(len(word) for word in matched_words)
    return words_length / domain_length if domain_length > 0 else 0.0
    
def dga_ngram_ratio(domain:str) -> float:
    """
    Function extracts 2,3,4-grams from the domain
    and look them up in dga n-gram database.
    For every n-gram (2,3,4) is calculated the ratio
    (found n-grams) / (all extracted n-grams from domain).
    After calculating the ratio for 2,3,4-grams we calculate
    the average across all n-grams ratio from previous step.
    (2-gram avg) + (3-gram avg) + (4-gram avg) / 3
    Args:
        domain (str): Domain name

    Returns:
        float: n-gram ratio
    """
    
    
def nondga_ngram_ratio(domain:str) -> float:
    """
    Function calculates the ratio same as
    'dga_ngram_ratio' but for n-gram matching is used
    non-dga ngram database

    Args:
        domain (str): Domain name

    Returns:
        float: n-gram ratio
    """

def visualize_roc_curve(y_test, y_pred):
    # Calculate fpr and tpr for different probability thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Calculate AUC score
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def extract_features_from_sld(domain: str, known_tlds: set, ngrams: N_grams):
    """
    Function extracts specified features and writes
    them to the pandas dataframe

    Args:
        domain (str): fully specified domain name
        known_tlds (set): Set of the known tlds
    """
    # - Extract subdomains from domains
    ext = tldextract.extract(domain)
    ext_subdomain = ext.subdomain
    ext_domain = ext.domain
    ext_suffix = ext.suffix

    if not ext_domain:
        #Parse suffix
        #How many words are delimited
        domain_count = ext_suffix.split(".")
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
    # 11. Dictionary matching ratio (matched_words)/(domain_len)
    dictionary_match = dictionary_matching(ext_domain)
    # 12. DGA n-gram ratio
    dga_ngram_ratio = ngrams.dga_ngram_avg_ratio(ext_domain)
    # 13. Non-DGA n-gram ratio
    nondga_ngram_ratio = ngrams.nondga_ngram_avg_ratio(ext_domain)
    # 14. Flag of beginning digit
    first_digit_flag = flag_of_first_digit(ext_domain)
    # 15. Flag of malicous domain -> Flag of well-known TLD NOPE for now
    well_known_tld = verify_tld(ext_suffix, known_tlds)
    # 16. Normalized entropy
    norm_entropy = get_normalized_entropy(ext_domain)
    # 17. Number of subdomains
    subdomains_count = count_subdomains(domain)
     # 18. Contains 'www'
    www_flag = contains_www(domain)    
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
        dictionary_match,
        dga_ngram_ratio,
        nondga_ngram_ratio,
        first_digit_flag,
        well_known_tld,
        norm_entropy,
        subdomains_count,
        www_flag
    )
    
def extract_features_from_concat(domain: str, known_tlds: set, ngrams: N_grams):
    """
    Function extracts specified features and writes
    them to the pandas dataframe from concatenated subdomains

    Args:
        domain (str): fully specified domain name
        known_tlds (set): Set of the known tlds
    """
    # - Extract subdomains from domains
    ext = tldextract.extract(domain)
    ext_subdomain = ext.subdomain
    ext_domain = ext.domain
    ext_suffix = ext.suffix
    concat_subdomains = remove_tld(domain).replace(".","")
    
    if not ext_domain:
        #Parse suffix
        #How many words are delimited
        domain_count = ext_suffix.split(".")
        if len(domain_count) > 1:
            ext_domain = domain_count[-2]
            concat_subdomains = domain_count[-2]
            ext_suffix = domain_count[-1]
            
    # 1. Domain length
    domain_length = domain_len(concat_subdomains)
    # 2. TLD length
    tld_len = domain_len(ext_suffix)
    # 3. SLD length
    sld_len = domain_len(ext_domain)
    # 4. The length of the longest consonant sequence
    max_consonant_len = longest_consonant_seq(concat_subdomains)
    # 5. The number of digits in the domain
    sld_digits_len = count_digits(concat_subdomains)
    # 6. The number of unique characters in the domain (SLD + TLD)
    unique_chars = unique_character_count(ext_domain + ext_suffix)
    # 7. Digit ratio
    try:
        digit_ratio = sld_digits_len / domain_length
    except ZeroDivisionError as err:
        print(err)
        print(domain)
        print(concat_subdomains)
    # 8. Consonant ratio
    consonant_ratio = get_consonant_ratio(concat_subdomains)
    # 9. Non-alfanumeric ratio
    non_alfa_ratio = get_non_alfa_numeric_ratio(concat_subdomains)
    # 10. Hexadecimal ratio
    hex_ratio = get_hex_ratio(concat_subdomains)
    # 11. Dictionary matching ratio (matched_words)/(domain_len)
    dictionary_match = dictionary_matching(concat_subdomains)
    # 12. DGA n-gram ratio
    dga_ngram_ratio = ngrams.dga_ngram_avg_ratio(concat_subdomains)
    # 13. Non-DGA n-gram ratio
    nondga_ngram_ratio = ngrams.nondga_ngram_avg_ratio(concat_subdomains)
    # 14. Flag of beginning digit
    first_digit_flag = flag_of_first_digit(concat_subdomains)
    # 15. Flag of malicous domain -> Flag of well-known TLD NOPE for now
    well_known_tld = verify_tld(ext_suffix, known_tlds)
    # 16. Normalized entropy
    norm_entropy = get_normalized_entropy(concat_subdomains)
    # 17. Number of subdomains
    subdomains_count = count_subdomains(domain)
    # 18. Contains 'www'
    www_flag = contains_www(domain)

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
        dictionary_match,
        dga_ngram_ratio,
        nondga_ngram_ratio,
        first_digit_flag,
        well_known_tld,
        norm_entropy,
        subdomains_count,
        www_flag)
    
def extract_features_averaged(domain: str, known_tlds: set, ngrams: N_grams):
    """
    Function extracts specified features and writes
    them to the pandas dataframe from averaged subdomains

    Args:
        domain (str): fully specified domain name
        known_tlds (set): Set of the known tlds
    """
    # - Extract subdomains from domains
    domain_labels = extract_subdomains(domain)
    ext = tldextract.extract(domain)
    ext_suffix = ext.suffix
    ext_domain = ext.domain
    
    if domain_labels == ['']:
        #Parse suffix
        #How many words are delimited
        domain_count = ext_suffix.split(".")
        if len(domain_count) > 1:
            domain_labels= [domain_count[-2]]
            ext_domain = domain_count[-2]
            ext_suffix = domain_count[-1]
    # 1. Domain length
    domain_length = np.array([domain_len(label) for label in domain_labels]).mean()
    # 2. TLD length
    tld_len = domain_len(ext_suffix)
    # 3. SLD length
    sld_len = domain_len(ext_domain)
    # 4. The length of the longest consonant sequence
    max_consonant_len = np.array([longest_consonant_seq(label)
                                  for label in domain_labels]).mean()
    # 5. The number of digits in the domain
    sld_digits_len = np.array([count_digits(label)
                               for label in domain_labels]).mean()
    # 6. The number of unique characters in the domain (SLD + TLD)
    unique_chars = unique_character_count(ext_domain + ext_suffix)
    # 7. Digit ratio
    digit_ratio = sld_digits_len / domain_length
    # 8. Consonant ratio
    consonant_ratio = np.array([get_consonant_ratio(label)
                                for label in domain_labels]).mean()
    # 9. Non-alfanumeric ratio
    try:
        non_alfa_ratio = np.array([get_non_alfa_numeric_ratio(label)
                                for label in domain_labels]).mean()
    except ZeroDivisionError as err:
        print(domain_labels)
        print(ext_domain)
        print(ext_suffix)
        print(err)
    # 10. Hexadecimal ratio
    hex_ratio = np.array([get_hex_ratio(label)
                                  for label in domain_labels]).mean()
    # 11. Dictionary matching ratio (matched_words)/(domain_len)
    dictionary_match = np.array([dictionary_matching(label)
                                  for label in domain_labels]).mean() 
    # 12. DGA n-gram ratio
    dga_ngram_ratio = np.array([ngrams.dga_ngram_avg_ratio(label)
                                  for label in domain_labels]).mean()
    # 13. Non-DGA n-gram ratio
    nondga_ngram_ratio = np.array([ngrams.nondga_ngram_avg_ratio(label)
                                  for label in domain_labels]).mean()
    # 14. Flag of beginning digit
    first_digit_flag = np.array([flag_of_first_digit(label)
                                  for label in domain_labels]).mean()
    # 15. Flag of malicous domain -> Flag of well-known TLD NOPE for now
    well_known_tld = verify_tld(ext_suffix, known_tlds)
    
    # 16. Normalized entropy
    norm_entropy = np.array([get_normalized_entropy(label)
                                for label in domain_labels]).mean()
    # 17. Number of subdomains
    subdomains_count = count_subdomains(domain)
    # 18. Contains 'www'
    www_flag = contains_www(domain)
    
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
        dictionary_match,
        dga_ngram_ratio,
        nondga_ngram_ratio,
        first_digit_flag,
        well_known_tld,
        norm_entropy,
        subdomains_count,
        www_flag)
    
def extract_features_4(domain: str, known_tlds: set, ngrams: N_grams):
    """
    Function extracts specified features and writes
    them to the pandas dataframe from averaged subdomains

    Args:
        domain (str): fully specified domain name
        known_tlds (set): Set of the known tlds
    """
    # - Extract subdomains from domains
    domain_labels = extract_subdomains(domain)
    ext = tldextract.extract(domain)
    ext_suffix = ext.suffix
    ext_domain = ext.domain
    
    if domain_labels == ['']:
        #Parse suffix
        #How many words are delimited
        domain_count = ext_suffix.split(".")
        print(domain_count)
        if len(domain_count) > 1:
            domain_labels= [domain_count[-2]]
            ext_domain = domain_count[-2]
            ext_suffix = domain_count[-1]
    
    pca = PCA(n_components=1)  # Specify the number of components to reduce to (in this case, 1)
    # 1. Domain length
    domain_length = np.array([domain_len(label) for label in domain_labels]).reshape(-1, 1)
    # Perform PCA on the array
    result = pca.fit_transform(domain_length)
    domain_length = result[0][0]

    # 2. TLD length
    tld_len = domain_len(ext_suffix)
    # 3. SLD length
    sld_len = domain_len(ext_domain)
    # 4. The length of the longest consonant sequence
    max_consonant_len = np.array([longest_consonant_seq(label)
                                  for label in domain_labels]).reshape(-1, 1)
    result = pca.fit_transform(max_consonant_len)
    max_consonant_len = result[0][0]
    # 5. The number of digits in the domain
    sld_digits_len = np.array([count_digits(label)
                               for label in domain_labels]).reshape(-1, 1)
    result = pca.fit_transform(sld_digits_len)
    sld_digits_len = result[0][0]
    # 6. The number of unique characters in the domain (SLD + TLD)
    unique_chars = unique_character_count(ext_domain + ext_suffix)
    # 7. Digit ratio
    digit_ratio = sld_digits_len / domain_length
    # 8. Consonant ratio
    consonant_ratio = np.array([get_consonant_ratio(label)
                                for label in domain_labels]).reshape(-1, 1)
    result = pca.fit_transform(consonant_ratio)
    consonant_ratio = result[0][0]
    # 9. Non-alfanumeric ratio
    try:
        non_alfa_ratio = np.array([get_non_alfa_numeric_ratio(label)
                                for label in domain_labels]).reshape(-1, 1)
        result = pca.fit_transform(non_alfa_ratio)
        non_alfa_ratio = result[0][0]
    except ZeroDivisionError as err:
        print(domain_labels)
        print(ext_domain)
        print(ext_suffix)
        print(err)
    # 10. Hexadecimal ratio
    hex_ratio = np.array([get_hex_ratio(label)
                                  for label in domain_labels]).reshape(-1, 1)
    result = pca.fit_transform(hex_ratio)
    hex_ratio = result[0][0]
    # 11. Flag of beginning digit
    first_digit_flag = flag_of_first_digit(ext_domain)
    # 12. Flag of malicous domain -> Flag of well-known TLD NOPE for now
    well_known_tld = verify_tld(ext_suffix, known_tlds)
    # 13. Normalized entropy
    norm_entropy = np.array([get_normalized_entropy(label)
                                  for label in domain_labels]).reshape(-1, 1)
    result = pca.fit_transform(norm_entropy)
    norm_entropy = result[0][0]
    # 14. Number of subdomains
    subdomains_count = count_subdomains(domain)
    # 15. Contains 'www'
    www_flag = contains_www(domain)
    
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
        well_known_tld,
        norm_entropy,
        subdomains_count,
        www_flag)

def extract_features_from_max(domain: str, known_tlds: set, ngrams: N_grams):
    """
    Function extracts specified features and writes
    them to the pandas dataframe from averaged subdomains

    Args:
        domain (str): fully specified domain name
        known_tlds (set): Set of the known tlds
    """
    # - Extract subdomains from domains
    domain_labels = extract_subdomains(domain)
    ext = tldextract.extract(domain)
    ext_suffix = ext.suffix
    ext_domain = ext.domain
    
    if domain_labels == ['']:
        #Parse suffix
        #How many words are delimited
        domain_count = ext_suffix.split(".")
        if len(domain_count) > 1:
            domain_labels= [domain_count[-2]]
            ext_domain = domain_count[-2]
            ext_suffix = domain_count[-1]
    # 1. Domain length
    domain_length = np.array([domain_len(label) for label in domain_labels]).max()
    # 2. TLD length
    tld_len = domain_len(ext_suffix)
    # 3. SLD length
    sld_len = domain_len(ext_domain)
    # 4. The length of the longest consonant sequence
    max_consonant_len = np.array([longest_consonant_seq(label)
                                  for label in domain_labels]).max()
    # 5. The number of digits in the domain
    sld_digits_len = np.array([count_digits(label)
                               for label in domain_labels]).max()
    # 6. The number of unique characters in the domain (SLD + TLD)
    unique_chars = unique_character_count(ext_domain + ext_suffix)
    # 7. Digit ratio
    digit_ratio = sld_digits_len / domain_length
    # 8. Consonant ratio
    consonant_ratio = np.array([get_consonant_ratio(label)
                                for label in domain_labels]).max()
    # 9. Non-alfanumeric ratio
    try:
        non_alfa_ratio = np.array([get_non_alfa_numeric_ratio(label)
                                for label in domain_labels]).max()
    except ZeroDivisionError as err:
        print(domain_labels)
        print(ext_domain)
        print(ext_suffix)
        print(err)
    # 10. Hexadecimal ratio
    hex_ratio = np.array([get_hex_ratio(label)
                                  for label in domain_labels]).max()
    # 11. Dictionary matching ratio (matched_words)/(domain_len)
    dictionary_match = np.array([dictionary_matching(label)
                                  for label in domain_labels]).max() 
    # 12. DGA n-gram ratio
    dga_ngram_ratio = np.array([ngrams.dga_ngram_avg_ratio(label)
                                  for label in domain_labels]).max()
    # 13. Non-DGA n-gram ratio
    nondga_ngram_ratio = np.array([ngrams.nondga_ngram_avg_ratio(label)
                                  for label in domain_labels]).max()
    # 14. Flag of beginning digit
    first_digit_flag = np.array([flag_of_first_digit(label)
                                  for label in domain_labels]).max()
    # 15. Flag of malicous domain -> Flag of well-known TLD NOPE for now
    well_known_tld = verify_tld(ext_suffix, known_tlds)
    
    # 16. Normalized entropy
    norm_entropy = np.array([get_normalized_entropy(label)
                                for label in domain_labels]).max()
    # 17. Number of subdomains
    subdomains_count = count_subdomains(domain)
    # 18. Contains 'www'
    www_flag = contains_www(domain)
    
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
        dictionary_match,
        dga_ngram_ratio,
        nondga_ngram_ratio,
        first_digit_flag,
        well_known_tld,
        norm_entropy,
        subdomains_count,
        www_flag)

def visualize_correlation_matrix(corr_matrix):
    # create a heatmap of the correlation matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax, annot_kws={"size": 16})

    # increase the font size of the labels and the correlation values
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.show()
    
def main():
    path = "/mnt/c/Work/Bachelors-thesis/Dataset/DGA/Fraunhofer-dataset/dgarchive_datasets/"
    path2 = "/mnt/c/Work/Bachelors-thesis/Dataset/Non-DGA/VUT-FIT/"
    file = "01-Proportion-pick-all.csv"
    file2 = "vut-fit.csv"
    
    df = pd.read_csv(path + file, names=["domain"])
    df2 = pd.read_csv(path2 + file2, names=["domain"])
    
    known_subdomain_path = "/mnt/c/Work/Bachelors-thesis/Dataset/Non-DGA/public_suffix_list.dat.txt"
    tlds = KnownTLD(known_subdomain_path)
    
    ngram_dir = "/mnt/d/VUT/Bachelor-thesis/05-Github/DGA-Classifier/04-Models/ngrams/"
    dga_ngram_csv = ngram_dir + "dga-ngram.csv"
    nondga_ngram_csv = ngram_dir + "non-dga-ngram.csv"
    ngrams = N_grams(dga_ngram_csv=dga_ngram_csv, nondga_ngram_csv=nondga_ngram_csv)
    
    known_tlds = tlds.get_tlds()
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
    # - Extract subdomains from domains
    process_subdomain = 4
    if process_subdomain == 0: 
        df[feature_columns] = df["domain"].apply(lambda x: extract_features_from_sld(x, known_tlds, ngrams)).apply(pd.Series)
        df2[feature_columns] = df2["domain"].apply(lambda x: extract_features_from_sld(x, known_tlds, ngrams)).apply(pd.Series)
    elif process_subdomain == 1: # String concatenation 
        df[feature_columns] = df["domain"].apply(lambda x: extract_features_from_concat(x, known_tlds, ngrams)).apply(pd.Series)
        df2[feature_columns] = df2["domain"].apply(lambda x: extract_features_from_concat(x, known_tlds, ngrams)).apply(pd.Series)
    elif process_subdomain == 2: # Mean calculating
        df[feature_columns] = df["domain"].apply(lambda x: extract_features_averaged(x, known_tlds, ngrams)).apply(pd.Series)
        df2[feature_columns] = df2["domain"].apply(lambda x: extract_features_averaged(x, known_tlds, ngrams)).apply(pd.Series)
    elif process_subdomain == 3: # PCA -> not working because of negative numbers
        df[feature_columns] = df["domain"].apply(lambda x: extract_features_4(x, known_tlds, ngrams)).apply(pd.Series)
        df2[feature_columns] = df2["domain"].apply(lambda x: extract_features_4(x, known_tlds, ngrams)).apply(pd.Series)
    else: # MAX values
        df[feature_columns] = df["domain"].apply(lambda x: extract_features_from_max(x, known_tlds, ngrams)).apply(pd.Series)
        df2[feature_columns] = df2["domain"].apply(lambda x: extract_features_from_max(x, known_tlds, ngrams)).apply(pd.Series)

    df["dga"] = 1
    df2["dga"] = 0
    
    final_df = pd.concat([df,df2], ignore_index=True)

    # CREATE XGBOOST MODEL
    # Split the data
    X = final_df[feature_columns]
    y = final_df["dga"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    #Correlation matrix
    corr_matrix = X.corr()
    visualize_correlation_matrix(corr_matrix=corr_matrix)
    # Train the XGBoost model
    # params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 1, 'n_estimators': 100}
    params = {'objective': 'binary:logistic',
        #   'tree_method': 'gpu_hist',
          'max_depth': 9,
          'eta': 0.1,
          'min_child_weight': 1,
        #   'sampling_method': 'gradient_based',
          'subsample': 0.1,
          'n_estimators': 190
    }
    model = xgb.XGBClassifier(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    xgb.plot_importance(model)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5)
    
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    print(f"Time of the model training: {minutes} min {seconds}sec")
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print('ROC AUC:', roc_auc)
    print('Scores:', scores.mean())

    visualize_roc_curve(y_test, y_pred)
    
    # Export model to file
    model_path = "./04-Models/binary_classification/01_xgb_binary_clf.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    