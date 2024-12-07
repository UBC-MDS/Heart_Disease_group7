import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import altair_ally as aly
import click

@click.command()

def main(): 

    combined_df = pd.read_csv('../data/combined_df_clean.csv')

    #same data cleaning codes
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

    train_df, test_df = train_test_split(combined_df, test_size=0.3, random_state=123)

    numerical_chart = aly.dist(train_df, color='label')
    numerical_chart.save('../results/numerical_chart.png', format='png')

    categorical_chart = aly.dist(train_df, dtype = 'category', color = 'label')
    categorical_chart.save('../results/categorical_chart.png', format='png')

    train_df.to_csv('../data/train_df.csv')
    test_df.to_csv('../data/test_df.csv')

# call main function 
if __name__ == "__main__":
    main() # pass any command line args to main here