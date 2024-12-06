import pandas as pd
import numpy as np
import click
import pandera as pa
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation

@click.command()
@click.option('--input', type=str)
@click.option('--output', type=str)


def main(input, output): 

    combined_df = pd.read_csv(input)
    
    ## --- 1. Correct data file format

    file_path = [file_path_hungarian, file_path_switzerland, file_path_cleveland, file_path_va]
    if False in [path.endswith('.data') for path in file_path]:
        print("Warning: The file extension is not .data")
    else:
        print("File is in the expected format.")

    ## --- 2. Correct column names
    #The script ensures that the column names in the combined_df DataFrame match a predefined list of expected column names (

    expected_names = set(columns)
    actual_names = set(combined_df.columns)
    if expected_names != actual_names:
        print(f"Warning: Column names do not match. Expected: {columns}, Found: {combined_df.columns.tolist()}")
    else:
        print("Column names are correct.")
    
    ## --- 3. No empty observations
    #This script checks whether the files specified by the paths in the list file_path all have the .data file extension

    empty_obs_schema = pa.DataFrameSchema(
    checks = [pa.Check(lambda df: ~(df.isna().all(axis = 1)).any(), error = "Empty rows found.")]
    )
    try:
        empty_obs_schema.validate(combined_df)
        print("No missing row found.")
    except pa.errors.SchemaError as a:
        print(f"Warning: There are {combined_df.isna().sum().sum()} missing values in dataset.")

    # call main function 
    if __name__ == "__main__":
        main() # pass any command line args to main here

    ## --- 4. Missingness not beyond expected threshold
    #The script checks each column in combined_df to see if the proportion of missing values exceeds 5%. If it does, a warning is printed. Otherwise, it confirms that the column's missing values are within acceptable limits.

    threshold = 0.05
    missing_prop = combined_df.isna().mean()
    for col, prop in missing_prop.items():
        if prop > threshold:
            print(f"Warning: There're too many missing values in column '{col}'.")
        else:
            print(f"Column '{col}' passed the test of missingness.")

    ## --- 5. Correct data types in each column
    #This script uses the pandera library to validate the combined_df DataFrame's columns against a predefined schema (column_type_schema).
    #If the columns match the expected data types, it prints a success message. If any columns don't match the expected types,
    #it prints a warning with details about the mismatch

    column_type_schema = pa.DataFrameSchema(
        {
            "age": pa.Column(pa.Int, nullable = True),
            "sex": pa.Column(pa.Int, nullable = True),
            "cp": pa.Column(pa.String, nullable = True),
            "trestbps": pa.Column(pa.Int, nullable = True),
            "chol": pa.Column(pa.Int, nullable = True),
            "fbs": pa.Column(pa.Int, nullable = True),
            "restecg": pa.Column(pa.String, nullable = True),
            "thalach": pa.Column(pa.Int, nullable = True),
            "exang": pa.Column(pa.String, nullable = True),
            "oldpeak": pa.Column(pa.Float, nullable = True),
            "slope": pa.Column(pa.String, nullable = True),
            "ca": pa.Column(pa.Float, nullable = True),
            "thal": pa.Column(pa.String, nullable = True),
            "label": pa.Column(pa.Int, nullable = True)
        }    
    )
    try:
        column_type_schema.validate(combined_df)
        print("All columns have correct data types.")
    except pa.errors.SchemaError as e:
        print(f"Warning: Validation failed: {e}")

    ## --- 6. No duplicate observations
    #This script uses the pandera library to validate the combined_df DataFrame for duplicate rows.
    #If duplicates are found, it raises a warning and displays the duplicate rows. If no duplicates are found, it confirms that there are no duplicates.

    duplicate_obs_schema = pa.DataFrameSchema(
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="There're duplicate rows")
        ]
    )
    try:
        duplicate_obs_schema.validate(combined_df)
        print("No duplicate rows found.")
    except pa.errors.SchemaError as e:
        duplicate_rows = combined_df[combined_df.duplicated(keep=False)]
        print(f"Warning: There're duplicate rows: \n{duplicate_rows}.")
    
    ## --- 7. No outlier or anomalous values
    #This script uses the pandera library to validate the values in the combined_df DataFrame against a defined schema (values_schema) to ensure the values meet specific criteria (e.g., value ranges, membership in predefined sets).
    #It performs this validation after converting the values in combined_df to float (if not NaN) to facilitate numeric checks.

    values_schema = pa.DataFrameSchema({
        "age": pa.Column(float, pa.Check.between(0, 120), nullable=True),
        "sex": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True), 
        "cp": pa.Column(float, pa.Check.isin([1.0, 2.0, 3.0, 4.0]), nullable=True), 
        "trestbps": pa.Column(float, pa.Check.between(20, 220), nullable=True),
        "chol": pa.Column(float, pa.Check.between(50, 800), nullable=True), 
        "fbs": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True), 
        "restecg": pa.Column(float, pa.Check.isin([0.0, 1.0, 2.0]), nullable=True),  
        "thalach": pa.Column(float, pa.Check.between(50, 240), nullable=True),  
        "exang":  pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True),  
        "oldpeak": pa.Column(float, pa.Check.between(0.0, 7.0), nullable=True),  
        "slope": pa.Column(float, pa.Check.isin([1.0, 2.0, 3.0]), nullable=True),  
        "ca": pa.Column(float, pa.Check.between(0, 4), nullable=True), 
        "thal": pa.Column(float, pa.Check.isin([3.0, 6.0, 7.0]), nullable=True),  
        "label": pa.Column(float, pa.Check.between(0.0, 4.0), nullable=True),  
    })
    replicate_df = combined_df.applymap(lambda x: float(x) if pd.notnull(x) else x)
    try:
        values_schema.validate(replicate_df, lazy = True)
        print("No outlier or anomalous value found.")
    except pa.errors.SchemaErrors as e:
        print(f"Warning: There're outlier or anomalous values.")
    
    ## --- 9. Target/response variable follows expected distribution
    #This check is useful for understanding the distribution of the target variable (label
    proportions = combined_df.label.value_counts(normalize=True)
    print("Class proportions are", proportions)
    print("Class proportions are as expected.")

    ## --- 10. No anomalous correlations between target variable and features variables

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    ds = Dataset(combined_df, label='label', cat_features=categorical_features)

    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=ds)
    check_feat_lab_corr.run(dataset=ds).show()
    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("The correlation between target and features variables exceeds the threshold.")
    else:
        print("Everything is fine.")

    ## --- 11. No anomalous correlations between features variables
    #This code performs a Feature-Feature Correlation check to ensure that there are no anomalous or excessively high correlations
    #(above a threshold of 0.9) between feature variables in a dataset.

    check_feat_feat_corr = FeatureFeatureCorrelation(threshold=0.9)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=ds)
    check_feat_feat_corr.run(dataset=ds).show()

    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("The correlation between features variables exceeds the threshold.")
    else:
        print("Everything is fine.")

# call main function 
if __name__ == "__main__":
    main() # pass any command line args to main here