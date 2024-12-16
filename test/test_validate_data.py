import pytest
import pandas as pd
import sys 
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.validate_data import check_empty_obs, check_missingness, check_duplicate_obs

# Test for check_empty_obs
def test_check_empty_obs_no_empty_rows(capfd):
    """Test check_empty_obs with no completely empty rows."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    check_empty_obs(df)
    captured = capfd.readouterr()
    assert "No missing row found." in captured.out

def test_check_empty_obs_with_empty_rows(capfd):
    """Test check_empty_obs with completely empty rows."""
    df = pd.DataFrame({'A': [1, None], 'B': [2, None]})
    with pytest.raises(Exception) as exc_info:
        check_empty_obs(df)
    assert "Empty rows found." in str(exc_info.value)

# Test for check_missingness
def test_check_missingness_below_threshold(capfd):
    """Test check_missingness when missing values are below the threshold."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, None]})
    check_missingness(df, threshold=0.5)
    captured = capfd.readouterr()
    assert "Column 'A' passed the test of missingness." in captured.out
    assert "Column 'B' passed the test of missingness." in captured.out

def test_check_missingness_above_threshold(capfd):
    """Test check_missingness when missing values exceed the threshold."""
    df = pd.DataFrame({'A': [1, None, None], 'B': [2, None, None]})
    check_missingness(df, threshold=0.3)
    captured = capfd.readouterr()
    assert "Warning: There're too many missing values in column 'A'." in captured.out
    assert "Warning: There're too many missing values in column 'B'." in captured.out

# Test for check_duplicate_obs
def test_check_duplicate_obs_no_duplicates(capfd):
    """Test check_duplicate_obs with no duplicate rows."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    check_duplicate_obs(df)
    captured = capfd.readouterr()
    assert "No duplicate rows found." in captured.out

def test_check_duplicate_obs_with_duplicates(capfd):
    """Test check_duplicate_obs with duplicate rows."""
    df = pd.DataFrame({'A': [1, 2, 2], 'B': [4, 5, 5]})
    
    # Call the function (no exception expected)
    check_duplicate_obs(df)
    
    # Capture printed output
    captured = capfd.readouterr()
    assert "Warning: There're duplicate rows" in captured.out