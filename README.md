# Heart_Disease_group7

Authors:

-   Alex Wong
-   Caroline Kahare
-   Ethan Fang

## About

The objective of this project is to analyze and model the heart disease dataset to uncover patterns and build a classification model that predicts the presence of heart disease based on key health metrics and attributes. By leveraging data-driven insights, this project aims to contribute to the understanding and early detection of heart disease, which is crucial for effective medical intervention and prevention.

The dataset is multivariate and falls within the domain of health and medicine, with a primary focus on classification tasks. It consists of 303 instances and 13 key attributes, which include both categorical and numerical variables. The target variable, num, indicates the presence or absence of heart disease, with values ranging from 0 (no presence) to 4 (indicating varying levels of severity). The dataset also contains demographic, clinical, and diagnostic attributes, offering a comprehensive view of patient health metrics.

Key features of the dataset include demographic indicators such as age (age in years) and sex (gender: 1 = male, 0 = female). Clinical measurements include trestbps (resting blood pressure), chol (serum cholesterol levels), thalach (maximum heart rate achieved), and oldpeak (ST depression induced by exercise). Additionally, categorical variables such as cp (chest pain type), fbs (fasting blood sugar \> 120 mg/dl), restecg (resting electrocardiographic results), exang (exercise-induced angina), slope (slope of the peak exercise ST segment), ca (number of major vessels colored by fluoroscopy), and thal (heart imaging defects) provide valuable context for predicting heart disease.

The project will begin with data preprocessing, which includes handling missing values in the columns, ca and thal attributes, encoding categorical variables, and scaling numerical features where necessary. Exploratory Data Analysis (EDA) will follow, with visualizations and statistical summaries to identify trends and relationships among variables. For instance, the relationship between cholesterol levels and heart disease prevalence will be examined.

Once the data is prepared, classification models will be developed. The dataset will be split into training and testing subsets to ensure robust evaluation, and performance metrics like accuracy, precision, recall, and F1-score will be used to assess the models. Feature importance analysis will also be conducted to identify the most significant predictors, such as chest pain type (cp), ST depression (oldpeak), and defects in heart imaging (thal).

To improve model performance, hyperparameter tuning will be carried out using techniques such as cross-validation. The final model will be tested on unseen data to ensure it generalizes well. Findings will be summarized in a detailed report or visualized through a dashboard, highlighting clinical implications and actionable insights.

This project aims to develop a reliable classification model for heart disease prediction while identifying key health metrics influencing its presence. It will provide actionable recommendations for early diagnosis and preventive measures. Potential challenges include managing class imbalances, handling missing data, and ensuring the clinical relevance of machine learning results. Addressing these issues effectively will contribute to the success of the project and its impact on healthcare.

## Report

The final report can be found [here](docs/final_report.html)

## Dependencies

-   [Docker](https://www.docker.com/)

-   [VS Code](https://code.visualstudio.com/download)

-   [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Usage

We are using a conda virtual environment so that our computational environment is reproducible.

To replicate our analysis:

-   Install Miniconda/Anaconda.
-   Clone the repository.
-   Run conda env create -f environment.yml to create the environment.
-   Activate the environment
-   Run the analysis scripts

### Setup

-   Clone the repository.

-   Navigate to the root of this project on your computer using the command line and enter 'docker compose up'

-   In the terminal, look for a URL that starts with `http://127.0.0.1:8888/lab?token=`. Copy and paste that URL into your browser.

-   To run the analysis, open a terminal and run the following commands

    ``` python
    python download_data.py

    python data_cleaning_script.py --input ../data/combined_df.csv --output ../data/combined_df_clean.csv

    python data_validation_script.py --input ../data/combined_df_clean.csv

    python EDA_script.py

    python preprocessor.py --data-folder ../data --output-folder ../data

    python models_and_results.py

    quarto render report/Heart_Disease.qmd --to html
    quarto render report/Heart_Disease.qmd --to pdf
    ```

-   <div>

    ### Clean up

    </div>

-   To shut down the container and clean up the resources, type `Cntrl` + `C` in the terminal where you launched the container, and then type `docker compose rm`To shut down the container and clean up the resources, type `Cntrl` + `C` in the terminal where you launched the container, and then type `docker compose rm`

## Developer notes

### Developer dependencies

-   `conda` (version 23.9.0 or higher)

-   `conda-lock` (version 2.5.7 or higher)

### Adding a new dependency

1.  Add the dependency to the `environment.yml` file on a new branch.

2.  Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the `conda-linux-64.lock` file.

3.  Re-build the Docker image locally to ensure it builds and runs properly.

4.  Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

5.  Update the `docker-compose.yml` file on your branch to use the new container image (make sure to update the tag specifically).

6.  Send a pull request to merge the changes into the `main` branch.

## License

The code of this project licensed under the terms of the MIT license.

## References

@misc{heart_disease_45, author = {Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert}, title = {{Heart Disease}}, year = {1989}, howpublished = {UCI Machine Learning Repository}, note = {{DOI}: <https://doi.org/10.24432/C52P4X>} }

@book{Python, author = {Van Rossum, Guido and Drake, Fred L.}, title = {Python 3 Reference Manual}, year = {2009}, isbn = {1441412697}, publisher = {CreateSpace}, address = {Scotts Valley, CA} }

@article{numpy, author = {Harris, Charles R and Millman, K Jarrod and van der Walt, St{'{e}}fan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J and Kern, Robert and Picus, Matti and Hoyer, Stephan and van Kerkwijk, Marten H and Brett, Matthew and Haldane, Allan and del R{'{i}}o, Jaime Fern{'{a}}ndez and Wiebe, Mark and Peterson, Pearu and G{'{e}}rard-Marchant, Pierre and Sheppard, Kevin and Reddy, Tyler and Weckesser, Warren and Abbasi, Hameer and Gohlke, Christoph and Oliphant, Travis E}, doi = {10.1038/s41586-020-2649-2}, issn = {1476-4687}, journal = {Nature}, number = {7825}, pages = {357--362}, title = {{Array programming with NumPy}}, url = {<https://doi.org/10.1038/s41586-020-2649-2>}, volume = {585}, year = {2020} }

@article{altair, title = {Altair: Interactive statistical visualizations for python}, author = {Jake VanderPlas}, doi = {10.21105/joss.01057}, issn = {2475-9066}, journal = {Journal of open source software}, number = {7825}, pages = {1057}, url = {<https://doi.org/10.21105/joss.01057>}, volume = {3}, issue = {32}, year = {2018} }

@article{scikit-learn, title={{Scikit-learn: Machine Learning in Python}}, author={ Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. }, journal={Journal of Machine Learning Research}, volume={12}, pages={2825--2830}, year={2011} }

@article{pandas, title={{pandas: Python Data Analysis Library}}, author={McKinney, Wes and others}, journal={Python for Data Analysis}, year={2012}, publisher={O'Reilly Media, Inc.} }
