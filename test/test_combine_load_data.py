import pytest
import sys 
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
import pandas as pd
from src.combine_load_data import combine_load_data

# Fixture to set up test data
@pytest.fixture
def test_data(tmp_path):
    # Create temporary CSV files
    file1 = tmp_path / "test_file1.csv"
    file2 = tmp_path / "test_file2.csv"

    # Write test data to file1
    file1.write_text("1,2,3\n4,5,6\n")
    # Write test data to file2
    file2.write_text("7,8,9\n10,11,12\n")

    # Return file paths as a list
    return [str(file1), str(file2)]

# Test case 1: Check correct loading and combination of files
def test_combine_load_data_success(test_data):
    # Define column names
    col_names = ['col1', 'col2', 'col3']
    
    # Call the function
    combined_df = combine_load_data(test_data, col_names)
    print(combined_df)
    # Expected DataFrame
    expected_data = {
        'col1': [1, 4, 7, 10],
        'col2': [2, 5, 8, 11],
        'col3': [3, 6, 9, 12]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Check if the combined DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(combined_df, expected_df)

# Test case 2: Check behavior when a file does not exist
def test_combine_load_data_file_not_found():
    # Provide an invalid file path
    file_paths = ['non_existent_file.csv']
    col_names = ['a', 'b', 'c']

    # Expect the function to raise a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        combine_load_data(file_paths, col_names)

# Test case 3: Check behavior with mismatched column names
def test_combine_load_data_column_mismatch(test_data):
    # Provide incorrect column names
    col_names = ['wrong_col1', 'wrong_col2', 'wrong_col3']

    # Call the function
    combined_df = combine_load_data(test_data, col_names)

    # Check if the column names in the DataFrame match the provided names
    assert list(combined_df.columns) == col_names

    
# pytest test/test_combine_load_data.py