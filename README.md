# DGA-Classifier

This repository contains Python code that implements DGA (Domain Generation Algorithm) classification for both binary and multiclass classification. The code utilizes different techniques for each type of classification.

## Binary Classification

For binary classification, XGboost is used to classify domain names as either benign or malicious. The code trains the XGboost model using a set of labeled data and uses it to predict the labels of unseen domain names. This approach has been shown to be effective in accurately identifying malicious domains.

## Multiclass Classification

For multiclass classification, the code uses a combination of regex matching and statistical counter to classify domain names into one of several categories. The code first applies regex matching to identify specific patterns in the domain names, such as the presence of certain characters or strings. It then uses a statistical counter to count the frequency of specific characters and strings in the domain names. Based on these features, the code classifies the domain names into different categories, such as malware, phishing, or spam.

## Usage

To use this code, simply clone the repository and run the main.py script. The script will prompt you to specify the type of classification you want to perform (binary or multiclass) and the data set you want to use. Once you've provided this information, the script will train the appropriate model and use it to classify the domain names in the specified data set.

## Conclusion
Overall, this repository provides a powerful tool for DGA classification that can be used to identify and classify malicious domain names. Whether you're working on cybersecurity research or need to analyze large sets of domain names, this code can help you quickly and accurately classify domain names for a variety of purposes.
