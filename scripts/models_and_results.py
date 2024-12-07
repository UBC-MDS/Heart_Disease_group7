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
    
    
    np.random.seed(seed)
    models = {
        'Logistic Regression': LogisticRegression(random_state = 123, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Machine': SVC(random_state = 123, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    param_distributions = {
        'Logistic Regression': {
            'logisticregression__C': stats.loguniform(1e-3, 1e3),
            'logisticregression__solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'decisiontreeclassifier__max_depth': [3, 5, 10],
            'decisiontreeclassifier__min_samples_split': stats.randint(2, 20)
        },
        'Support Vector Machine': {
            'svc__C': stats.loguniform(1e-2, 1e2),
            'svc__kernel': ['linear', 'rbf']
        },
        'K-Nearest Neighbors': {
            'kneighborsclassifier__n_neighbors': stats.randint(3, 20),
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    }
    
    #RandomizedSearchCV allows for searching over a large hyperparameter space by sampling random values,
    #making it efficient compared to grid search.
    
    #Classification reports and confusion matrices provide insight into model performance,
    #including how well the model distinguishes between classes.
    best_models = {}

    best_models = {}
    
    for model_name, model in models.items():
        print(f"Tuning hyperparameters for {model_name} using RandomizedSearchCV...")
        
        clf_pipe = make_pipeline(preprocessor, model)
        
        random_search = RandomizedSearchCV(
            estimator=clf_pipe,
            param_distributions=param_distributions[model_name],
            scoring="accuracy",
            n_iter=10, 
            cv=5,
            random_state=seed
        )
        
        random_search.fit(X_train, y_train)
        
        best_models[model_name] = random_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        print("-" * 40)
    
    for model_name, model in best_models.items():
        print(f"Evaluating {model_name} on test set...")
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred,zero_division=0)
        print("Classification Report:")
        print(report)
        output_file = f"../results/classification_report_{model_name}.txt"
        with open(output_file, "w") as f:
            f.write(report)

if __name__ == '__main__':
    main()