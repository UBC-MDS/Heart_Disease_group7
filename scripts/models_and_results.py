# heart_disease_models_and_results
# author: Alex Wong, Caroline Kahare, Ethan Fang
# date: 2024-12-04

import click
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import sys 
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.models_fit_and_result_output import models_fit_and_result_output

@click.command()
@click.option('--output-file-path', type=str, help='Path to the output results file', default="results/model_evaluation_results.txt")
@click.option('--seed', type=int, help='Random seed for reproducibility', default=123)
def main(output_file_path, seed):
    
    #The models dictionary holds the classifier objects for different algorithms.
    #The param_distributions dictionary specifies the ranges and values for hyperparameters to be explored during optimization.
    #This setup allows for an efficient search over multiple hyperparameters and algorithms to find the best configuration for the task at hand.

    X_train = pd.read_csv('../data/x_train.csv', index_col = 0)
    y_train = pd.read_csv('../data/y_train.csv')
    X_test = pd.read_csv('../data/x_test.csv', index_col = 0)
    y_test = pd.read_csv('../data/y_test.csv')
    y_train = y_train['label']
    y_test = y_test['label']
    preprocessor = pickle.load(open('../data/preprocessor.pkl', "rb"))
    
    output_file_dir = "results"
    models_fit_and_result_output(preprocessor, X_train, y_train, X_test, y_test, output_file_dir,seed=123)

if __name__ == '__main__':
    main()