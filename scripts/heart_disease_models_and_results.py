# heart_disease_models_and_results
# author: Alex Wong, Caroline Kahare, Ethan Fang
# date: 2024-12-04

import click
import numpy as np
from sklearn.pipeline import Pipeline
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

@click.command()
@click.option('--X-train', type=str, help='Path to the input X train dataset file', required=True)
@click.option('--X-test', type=str, help='Path to the input X test dataset file', required=True)
@click.option('--y-train', type=str, help='Path to the input y train dataset file', required=True)
@click.option('--y-test', type=str, help='Path to the input y test dataset file', required=True)
@click.option('--output-file-path', type=str, help='Path to the output results file', default="results/model_evaluation_results.txt")
@click.option('--seed', type=int, help='Random seed for reproducibility', default=42)
def main(X_train, X_test, y_train, y_test, output_file_path, seed):
    
    #The models dictionary holds the classifier objects for different algorithms.
    #The param_distributions dictionary specifies the ranges and values for hyperparameters to be explored during optimization.
    #This setup allows for an efficient search over multiple hyperparameters and algorithms to find the best configuration for the task at hand.
    np.random.seed(seed)
    models = {
        'Logistic Regression': LogisticRegression(random_state = 123, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Machine': SVC(random_state = 123, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    param_distributions = {
        'Logistic Regression': {
            'classifier__C': stats.loguniform(1e-3, 1e3),
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'classifier__max_depth': [3, 5, 10],
            'classifier__min_samples_split': stats.randint(2, 20)
        },
        'Support Vector Machine': {
            'classifier__C': stats.loguniform(1e-2, 1e2),
            'classifier__kernel': ['linear', 'rbf']
        },
        'K-Nearest Neighbors': {
            'classifier__n_neighbors': stats.randint(3, 20),
            'classifier__weights': ['uniform', 'distance']
        }
    }
    
    #RandomizedSearchCV allows for searching over a large hyperparameter space by sampling random values,
    #making it efficient compared to grid search.
    
    #Classification reports and confusion matrices provide insight into model performance,
    #including how well the model distinguishes between classes.
    best_models = {}

    with open(output_file_path, "w") as f:  
        for model_name, model in models.items():
            f.write(f"Tuning hyperparameters for {model_name} using RandomizedSearchCV...\n")
            clf = Pipeline(steps=[('classifier', model)])
            
            random_search = RandomizedSearchCV(
                estimator=clf,
                param_distributions=param_distributions[model_name],
                scoring=make_scorer(roc_auc_score, needs_proba=True),
                n_iter=10, 
                cv=5,
                random_state=42
            )
            
            random_search.fit(X_train, y_train)
            
            best_models[model_name] = random_search.best_estimator_
            
            f.write(f"Best parameters for {model_name}: {random_search.best_params_}\n")
            f.write("-" * 40 + "\n")
        
        for model_name, model in best_models.items():
            f.write(f"Evaluating {model_name} on test set...\n")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)
            
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion) + "\n")
            f.write("-" * 40 + "\n")

if __name__ == '__main__':
    main()