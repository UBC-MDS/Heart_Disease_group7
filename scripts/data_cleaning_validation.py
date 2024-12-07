import pandas as pd
import numpy as np
import click

@click.command()
@click.option('--path', type=str)

def main(path): 

    combined_df = pd.read_csv(path)
    combined_df.replace("?", np.nan, inplace = True)

    columns = ['age','sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    

    ## delete duplicated rows
    #This process ensures the data is clean, well-formatted, and appropriate for analysis or modeling.
    duplicate_index = [102,187]
    combined_df = combined_df.drop(index=duplicate_index).reset_index(drop=True)

    # Casting continuous features to float64 instead of categories 
    combined_df['trestbps'] = combined_df['trestbps'].astype('float64')
    combined_df['chol'] = combined_df['chol'].astype('float64')
    combined_df['thalach'] = combined_df['thalach'].astype('float64')
    combined_df['oldpeak'] = combined_df['thalach'].astype('float64')

    # Casting label as categorical 
    combined_df['label'] = combined_df['label'].astype('category')

    # Casting as categorical 
    combined_df['cp'] = combined_df['cp'].astype('category')
    combined_df['sex'] = combined_df['sex'].astype('category')

    # For the following features, have to first convert dtype to number first to ensure the category labels
    # are not affected by decimals (i.e. 1.0 and 1 are not treated as different groups)
    combined_df['exang'] = pd.to_numeric(combined_df['exang'], errors='coerce').astype('category')
    combined_df['thal'] = pd.to_numeric(combined_df['thal'], errors='coerce').astype('category')
    combined_df['fbs'] = pd.to_numeric(combined_df['fbs'], errors='coerce').astype('category')
    combined_df['ca'] = pd.to_numeric(combined_df['ca'], errors='coerce').astype('category')
    combined_df['slope'] = pd.to_numeric(combined_df['slope'], errors='coerce').astype('category')
    combined_df['restecg'] = pd.to_numeric(combined_df['restecg'], errors='coerce').astype('category')


    combined_df.to_csv(path, index=False)


# call main function 
if __name__ == "__main__":
    main() # pass any command line args to main here