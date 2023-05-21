# DGA-Classifier

This repository contains Python code that implements DGA (Domain Generation Algorithm) classification for both binary and multiclass classification. The code utilizes different techniques for each type of classification.

## Binary Classification

For binary classification, XGBoost is used to classify domain names as either benign or malicious. The code trains the XGBoost model using a set of labeled data and uses it to predict the labels of unseen domain names. This approach has been shown to be effective in accurately identifying malicious domains.

## Multiclass Classification

For multiclass classification, the code uses a combination of regex matching and XGBoost multiclass model to classify domain names into one of 93 DGA families. The code first applies XGBoost model for the first classification. After XGBoost predicting the DGA family, the prediction is then checked with regex matching. One regex can match multiple families so it serves as a control mechanism for XGBoost model.





## Installation

To use this repository, you have to install requirements.txt. It's recommended to install requirements into virtual environment to avoid possible dependency conflicts.
In the description below there is a description of installation in a virtual environment

    conda create --name dga-env
    conda activate dga-env
    conda install pip
    pip install -r requirements.txt

## Documentation
The DGA Detector repository is organized into the following structure:

* **data**: This folder contains all the data used for training and testing the models. It includes both legitimate domain names and DGA-generated domain names, raw and also preprocessed data.

* **preprocessing**: The preprocessing folder contains scripts or modules responsible for data preprocessing tasks such as cleaning, tokenization, and feature extraction. This step prepares the data for training the models. One of the most important module is *dataset_paths.py* which contains all neccessary path for loading and storing trained models

* **training**: In the training folder, you will find scripts or modules that train various classification models using the preprocessed data. This step involves selecting appropriate machine learning algorithms, configuring hyperparameters, and fitting the models to the training data.

* **models**: Once the models are trained, they are saved in the models folder. This folder stores the serialized versions of the trained models along with any additional configurations or metadata required for later use. If you would like to save the models to a different folder you should edit paths in the *model_paths.py* implemented in 03-Models-training directory.

* **classifiers**: The classifiers folder contains scripts or modules that implement the binary and multiclass classification using the trained models from the models folder. The main classifier is *dga_classifier.py*, which consolidates binary and multiclass classifiers. The input to the classifier can be passed via _stdin_without specifying any parameters or you can use the classifier following manner:

        dga_classifier.py --domain="facebook.com"

* **tests**: The tests folder contains scripts or modules to evaluate the performance of the trained models. It includes test cases and evaluation metrics to assess the accuracy and effectiveness of the models.

## Attention
**All scripts in the repository have to be executed from project root directory because of the dependencies.** For example:

        python3 05-Classifiers/dga_classifier.py --domain="facebook.com"


## Conclusion

Overall, this repository provides a powerful tool for DGA classification that can be used to identify and classify malicious domain names. Whether you're working on cybersecurity research or need to analyze large sets of domain names, this code can help you quickly and accurately classify domain names for a variety of purposes.