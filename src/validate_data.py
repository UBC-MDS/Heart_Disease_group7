import pandas as pd
import numpy as np
import click
import pandera as pa
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation

def check_empty_obs(combined_df):
    """
    Check for completely empty rows in the DataFrame.

    This function uses `pandera` to validate if the input DataFrame contains 
    any rows where all values are missing (NaN). It raises a warning or an 
    error if such rows are found.

    Parameters:
        combined_df (pandas.DataFrame): The input DataFrame to be validated.
        
    Raises:
        pandera.errors.SchemaError: If the DataFrame contains completely empty rows.
        
    Prints:
        - "No missing row found." if there are no completely empty rows.
        - Warning message with the count of missing rows if completely empty rows are found.
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, None], 'B': [2, None]})
        >>> check_empty_obs(df)
        Warning: There are 1.0 missing values in dataset.
    """
    empty_obs_schema = pa.DataFrameSchema(
    checks = [pa.Check(lambda df: ~(df.isna().all(axis = 1)).any(), error = "Empty rows found.")]
    )
    try:
        empty_obs_schema.validate(combined_df)
        print("No missing row found.")
    except pa.errors.SchemaError as a:
        print(f"Warning: There are {combined_df.isna().sum().sum()} missing values in dataset.")
        raise a


def check_missingness(combined_df, threshold=0.05):
    """
    Check for missing values in the DataFrame and compare them to a threshold.

    This function calculates the proportion of missing values in each column
    and compares it to a user-defined threshold. It prints a warning for columns
    exceeding the threshold and passes columns with acceptable missingness.

    Parameters:
        combined_df (pandas.DataFrame): The input DataFrame to be checked for missing values.
        threshold (float, optional): The proportion of missing values in a column that is 
                                      considered acceptable. Defaults to 0.05.
        
    Prints:
        - A warning message for columns where the proportion of missing values exceeds the threshold.
        - A success message for columns where the proportion of missing values is below the threshold.
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [2, 2, None]})
        >>> check_missingness(df, threshold=0.3)
        Column 'A' passed the test of missingness.
        Warning: There're too many missing values in column 'B.'
    """
    missing_prop = combined_df.isna().mean()
    for col, prop in missing_prop.items():
        if prop > threshold:
            print(f"Warning: There're too many missing values in column '{col}'.")
        else:
            print(f"Column '{col}' passed the test of missingness.")


def check_duplicate_obs(combined_df):
    """
    Check for duplicate rows in the DataFrame.

    This function uses `pandera` to validate if the input DataFrame contains
    duplicate rows. It prints a success message if no duplicates are found,
    or a warning message showing the duplicate rows.

    Parameters:
        combined_df (pandas.DataFrame): The input DataFrame to be checked for duplicate rows.
        
    Raises:
        pandera.errors.SchemaError: If duplicate rows are found in the DataFrame.
        
    Prints:
        - "No duplicate rows found." if no duplicate rows exist.
        - Warning message with details of the duplicate rows if duplicates are found.
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': [2, 3, 3]})
        >>> check_duplicate_obs(df)
        Warning: There's duplicate rows:
           A  B
        1  2  3
        2  2  3
    """
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