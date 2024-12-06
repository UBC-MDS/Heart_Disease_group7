import os
import click
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

@click.command()
@click.option('--data-folder', default='data', help='Path to the folder containing input data files.')
@click.option('--output-folder', default='output', help='Path to the folder for saving output files.')
def preprocess(data_folder, output_folder):
    """Preprocess the train and test datasets and save the results."""
    # Input file paths
    train_file = os.path.join(data_folder, 'train_df.csv')
    test_file = os.path.join(data_folder, 'test_df.csv')

    # Output file paths
    os.makedirs(output_folder, exist_ok=True)
    preprocessor_file = os.path.join(output_folder, 'preprocessor.pkl')
    processed_train_file = os.path.join(output_folder, 'processed_X_train.csv')
    processed_test_file = os.path.join(output_folder, 'processed_X_test.csv')
    x_train_file = os.path.join(output_folder, 'x_train.csv')
    x_test_file = os.path.join(output_folder, 'x_test.csv')
    y_train_file = os.path.join(output_folder, 'y_train.csv')
    y_test_file = os.path.join(output_folder, 'y_test.csv')

    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Separate features and labels
    X_train = train_df.drop(columns=["label"])
    X_test = test_df.drop(columns=["label"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    # Save y_train and y_test
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)

    # Define feature groups
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['cp', 'restecg']
    binary_features = ['sex', 'exang', 'fbs']
    drop_features = ['thal', 'ca', 'slope']

    # Define transformation pipelines
    numeric_transformer_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    categorical_transformer_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(drop='if_binary', sparse_output=False)
    )
    binary_transformer = SimpleImputer(strategy='most_frequent')

    # Combine into a column transformer
    preprocessor = make_column_transformer(
        (numeric_transformer_pipe, numeric_features),
        (categorical_transformer_pipe, categorical_features),
        (binary_transformer, binary_features),
        ("drop", drop_features)
    )

    # Fit and transform the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get transformed column names
    col_names = (
        numeric_features +
        preprocessor.named_transformers_['pipeline-1'].get_feature_names_out().tolist() + 
        binary_features
    )

    # Convert transformed data into DataFrames
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=col_names)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=col_names)

    # Save the processed data
    X_train_transformed.to_csv(processed_train_file, index=False)
    X_test_transformed.to_csv(processed_test_file, index=False)
    X_train.to_csv(x_train_file, index=False)
    X_test.to_csv(x_test_file, index=False)

    # Save the preprocessor
    with open(preprocessor_file, 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"Preprocessing complete. Files saved in {output_folder}.")

if __name__ == '__main__':
    preprocess()
