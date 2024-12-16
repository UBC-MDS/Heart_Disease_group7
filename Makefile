.PHONY: all clean


all: data/x_train.csv data/y_train.csv reports/analysis.html results/categorical_chart.png results/numerical_chart.png

# downloading data 
data/combined_df.csv: scripts/download_data.py
	python scripts/download_data.py

# data cleaning
data/combined_df_clean.csv: scripts/data_cleaning_script.py data/combined_df.csv
	python scripts/data_cleaning_script.py \
		--input data/combined_df.csv \
		--output data/combined_df_clean.csv

## EDA 
data/test_df.csv data/train_df.csv results/categorical_chart.png results/numerical_chart.png: scripts/EDA_script.py data/combined_df_clean.csv
	python scripts/EDA_script.py

# preprocessor 
data/preprocessor.pkl data/processed_X_test.csv data/processed_X_train.csv data/x_test.csv data/x_train.csv data/y_test.csv data/y_train.csv: scripts/preprocessor.py data/test_df.csv data/train_df.csv
	python scripts/preprocessor.py \
		--data-folder data \
		--output-folder data

# render report 
reports/analysis.html: results/numerical_chart.png results/categorical_chart.png reports/analysis.qmd
	quarto render reports/analysis.qmd

clean:
	rm -f data/combined_df.csv \
          data/combined_df_clean.csv \
          data/test_df.csv \
          data/train_df.csv \
          results/numerical_chart.png \
          results/categorical_chart.png \
          data/processed_X_test.csv \
          data/processed_X_train.csv \
          data/preprocessor.pkl \
          data/x_train.csv \
          data/x_test.csv \
          data/y_test.csv \
          data/y_train.csv \
          reports/analysis.html
          reports/analysis.html