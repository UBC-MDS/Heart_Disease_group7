# Heart_Disease_group7

Authors:

* Alex Wong
* Caroline Kahare
* Ethan Fang

## About

The objective of this project is to analyze and model the heart disease dataset to uncover patterns and build a classification model that predicts the presence of heart disease based on key health metrics and attributes. By leveraging data-driven insights, this project aims to contribute to the understanding and early detection of heart disease, which is crucial for effective medical intervention and prevention.

The dataset is multivariate and falls within the domain of health and medicine, with a primary focus on classification tasks. It consists of 303 instances and 13 key attributes, which include both categorical and numerical variables. The target variable, num, indicates the presence or absence of heart disease, with values ranging from 0 (no presence) to 4 (indicating varying levels of severity). The dataset also contains demographic, clinical, and diagnostic attributes, offering a comprehensive view of patient health metrics.

Key features of the dataset include demographic indicators such as age (age in years) and sex (gender: 1 = male, 0 = female). Clinical measurements include trestbps (resting blood pressure), chol (serum cholesterol levels), thalach (maximum heart rate achieved), and oldpeak (ST depression induced by exercise). Additionally, categorical variables such as cp (chest pain type), fbs (fasting blood sugar > 120 mg/dl), restecg (resting electrocardiographic results), exang (exercise-induced angina), slope (slope of the peak exercise ST segment), ca (number of major vessels colored by fluoroscopy), and thal (heart imaging defects) provide valuable context for predicting heart disease.

The project will begin with data preprocessing, which includes handling missing values in the columns, ca and thal attributes, encoding categorical variables, and scaling numerical features where necessary. Exploratory Data Analysis (EDA) will follow, with visualizations and statistical summaries to identify trends and relationships among variables. For instance, the relationship between cholesterol levels and heart disease prevalence will be examined.

Once the data is prepared, classification models will be developed. The dataset will be split into training and testing subsets to ensure robust evaluation, and performance metrics like accuracy, precision, recall, and F1-score will be used to assess the models. Feature importance analysis will also be conducted to identify the most significant predictors, such as chest pain type (cp), ST depression (oldpeak), and defects in heart imaging (thal).

To improve model performance, hyperparameter tuning will be carried out using techniques such as cross-validation. The final model will be tested on unseen data to ensure it generalizes well. Findings will be summarized in a detailed report or visualized through a dashboard, highlighting clinical implications and actionable insights.

This project aims to develop a reliable classification model for heart disease prediction while identifying key health metrics influencing its presence. It will provide actionable recommendations for early diagnosis and preventive measures. Potential challenges include managing class imbalances, handling missing data, and ensuring the clinical relevance of machine learning results. Addressing these issues effectively will contribute to the success of the project and its impact on healthcare.


## Report
The final report can be found [here](docs/final_report.html)

## Usage
We are using a conda virtual environment so that our computational environment is reproducible.

To replicate our analysis:

* Install Miniconda/Anaconda.
* Clone the repository.
* Run conda env create -f environment.yml to create the environment.
* Activate the environment
* Run the analysis scripts

## License
The code of this project licensed under the terms of the MIT license. 