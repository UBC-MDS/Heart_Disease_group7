import pandas as pd
import click

def combine_files(output_file):
    file_paths = [
        '../data/processed.hungarian.data',
        '../data/processed.switzerland.data',
        '../data/processed.cleveland.data',
        '../data/processed.va.data'
    ]
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    
    #read file into dataframes
    dataframes = [pd.read_csv(fp, index_col=False, names=columns) for fp in file_paths]
    
    #put all the dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    #save combined df
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

if __name__ == '__main__':
    combine_files('../data/combined_df.csv')