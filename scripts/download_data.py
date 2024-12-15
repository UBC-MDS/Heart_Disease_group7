import pandas as pd
import click
import sys 
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.combine_load_data import combine_load_data

def combine_files(output_file):
    file_paths = [
        '../data/processed.hungarian.data',
        '../data/processed.switzerland.data',
        '../data/processed.cleveland.data',
        '../data/processed.va.data'
    ]
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    
    combined_df = combine_load_data(file_paths, columns)
    
    #save combined df
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

if __name__ == '__main__':
    combine_files('../data/combined_df.csv')