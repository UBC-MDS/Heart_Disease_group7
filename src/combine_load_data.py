import pandas as pd

def combine_load_data(file_paths, col_names):
    """
    Combine and load data from multiple CSV files into a single DataFrame.
    
    This function reads multiple CSV files specified in the `file_paths` list, 
    applies column names provided in `col_names`, and combines them into a 
    single pandas DataFrame. It also handles the case where any file is not found.
    
    Parameters:
        file_paths (list of str): A list containing file paths to CSV files.
        col_names (list of str): A list of column names to apply to the loaded data.
        
    Returns:
        pandas.DataFrame: A single DataFrame that combines data from all input CSV files.
        
    Raises:
        FileNotFoundError: If any of the specified file paths cannot be found.
        
    Examples:
        >>> file_paths = ['data/file1.csv', 'data/file2.csv']
        >>> col_names = ['column1', 'column2', 'column3']
        >>> df = combine_load_data(file_paths, col_names)
        >>> print(df.head())
        
        This will load and combine 'file1.csv' and 'file2.csv', applying column names
        ['column1', 'column2', 'column3'].
    """
    try:
        dataframes = [pd.read_csv(fp, index_col=False, names=col_names) for fp in file_paths]
        combined_df = pd.concat(dataframes, ignore_index=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

    return combined_df