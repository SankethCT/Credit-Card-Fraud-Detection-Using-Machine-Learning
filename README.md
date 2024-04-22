# Credit Card Fraud Detection Project

## Overview

This project aims to detect fraudulent transactions in credit card data using machine learning techniques. The dataset used contains transactions made by credit cards in September 2013 by European cardholders. It contains a total of 284,807 transactions, with 492 frauds, making it highly imbalanced.

## Getting Started

### Dependencies

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- imbalanced-learn

### Installing

1. Clone the repository:
https://github.com/SankethCT/Credit-Card-Fraud-Detection-Using-Machine-Learning.git

2. Navigate to the project directory:
cd credit_card_fraud_detection


3. Install the required Python packages using pip:

    Python 3.x
    Jupyter Notebook
    scikit-learn
    pandas
    numpy
    matplotlib
    imbalanced-learn


### Executing Program

1. Open a Jupyter Notebook:

2. Open the `credit_card_fraud_detection.ipynb` notebook.

3. Execute each cell in the notebook step by step.

## Dataset

The dataset contains the following columns:

- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: Principal components obtained with PCA. Features are anonymized to protect sensitive information.
- **Amount**: Transaction amount.
- **Class**: The target variable indicating whether the transaction is fraudulent (1) or genuine (0).

## Exploratory Data Analysis (EDA)

- The dataset does not contain any missing values.
- The majority of transactions are genuine (99.83%), with only a small percentage of fraudulent transactions (0.17%).
- Visualizations of the class distribution show a severe class imbalance.

## Model Building

Three machine learning models were trained on the original dataset:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Naive Bayes Classifier**

### Evaluation Metrics

For each model, the following evaluation metrics were calculated on the test set:

- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positive instances.
- **F1-score**: The harmonic mean of precision and recall.

## Addressing Class Imbalance

Due to the severe class imbalance, the dataset was resampled using the Synthetic Minority Oversampling Technique (SMOTE) to balance the classes. The resampled dataset contains an equal number of fraudulent and genuine transactions.

The same machine learning models were trained on the resampled dataset, and their performance was evaluated using the same metrics.

## Results

### Original Dataset:

- **Decision Tree**:
  - Accuracy: 99.91%
  - Precision: 69.80%
  - Recall: 76.47%
  - F1-score: 72.98%

- **Random Forest**:
  - Accuracy: 99.96%
  - Precision: 94.02%
  - Recall: 80.88%
  - F1-score: 86.96%

- **Naive Bayes**:
  - Accuracy: 97.81%
  - Precision: 5.85%
  - Recall: 84.56%
  - F1-score: 10.94%

### Resampled Dataset (SMOTE):

- **Decision Tree**:
  - Accuracy: 99.80%
  - Precision: 99.71%
  - Recall: 99.99%
  - F1-score: 99.88%

- **Random Forest**:
  - Accuracy: 99.99%
  - Precision: 99.98%
  - Recall: 99.99%
  - F1-score: 99.88%

- **Naive Bayes**:
  - Accuracy: 91.49%
  - Precision: 97.17%
  - Recall: 85.49%
  - F1-score: 90.96%

## Conclusion

- Random Forest performed the best on the original dataset, but its performance significantly improved after addressing class imbalance.
- SMOTE oversampling technique effectively balanced the classes, resulting in improved performance for all models.
- Naive Bayes, despite its simplicity, showed considerable improvement after oversampling but still lagged behind the tree-based models.


