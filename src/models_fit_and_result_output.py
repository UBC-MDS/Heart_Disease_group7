import click
import os
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

def models_fit_and_result_output(preprocessor,X_train,y_train,X_test,y_test,output_file_dir,seed=999):
    """
    Train and evaluate multiple machine learning models with hyperparameter tuning.

    This function trains multiple models using a pipeline that includes a specified preprocessor
    and performs hyperparameter tuning using `RandomizedSearchCV`. The best model for each 
    algorithm is evaluated on the test set, and classification reports are generated and 
    saved to the specified output file.

    Parameters:
        preprocessor (sklearn.pipeline.Pipeline): A preprocessing pipeline to be applied 
                                                  to the training and test data.
        X_train (pandas.DataFrame or numpy.ndarray): Feature data for training.
        y_train (pandas.Series or numpy.ndarray): Target labels for training.
        X_test (pandas.DataFrame or numpy.ndarray): Feature data for testing.
        y_test (pandas.Series or numpy.ndarray): Target labels for testing.
        output_file_path (str): File path to save the classification reports for all models.

    Models:
        - Logistic Regression
        - Decision Tree
        - Support Vector Machine (SVM)
        - k-Nearest Neighbors (k-NN)
    
    Hyperparameter Search:
        - Hyperparameters for each model are defined in the `param_distributions` dictionary.
        - RandomizedSearchCV is used to search over the hyperparameter space.

    Output:
        - Prints the best hyperparameters for each model after hyperparameter tuning.
        - Prints and saves classification reports for each model's performance on the test set.
        - Saves the classification reports to the specified output file.

    Returns:
        None
    
    Raises:
        FileNotFoundError: If the specified `output_file_path` is invalid.

    Examples:
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.compose import ColumnTransformer
        >>> import pandas as pd

        # Load data
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target

        # Train-test split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessor
        >>> preprocessor = ColumnTransformer([('scaler', StandardScaler(), X.columns)], remainder='passthrough')

        # Run function
        >>> models_fit_and_output(preprocessor, X_train, y_train, X_test, y_test, 'output_reports.txt')

    Notes:
        - The function assumes that `RandomizedSearchCV` is suitable for the problem at hand.
        - It uses `accuracy` as the scoring metric during hyperparameter tuning.
        - The random seed ensures reproducibility.
    """
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
        output_file = os.path.join(output_file_dir, f"classification_report_{model_name.replace(' ', '_')}.txt")
        with open(output_file, "w") as f:
            f.write(report)