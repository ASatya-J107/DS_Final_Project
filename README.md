# Exam Score Prediction

## Overview

This repository contains a data analysis and machine learning project that aims to predict Math, Reading, and Writing scores based on various features. The project is deployed and accessible through the web application hosted at [https://exam-score-prediction.streamlit.app/](https://exam-score-prediction.streamlit.app/). This README provides an overview of the project structure, the steps involved in the analysis and modeling process, and instructions for accessing the deployed project.

## Project Structure

The project is organized into several main steps, each of which is explained below:

### 1. Importing Libraries

In this step, we import various Python libraries commonly used for data analysis and machine learning tasks. These libraries include:

- NumPy
- Matplotlib
- Pandas
- Seaborn
- Scikit-Learn
- XGBoost
- Other necessary libraries

### 2. Importing Dataset

We load a dataset from a URL using Pandas and store it in a variable called `dataset`. We also display basic information about the dataset using `dataset.info()` and show the first few rows using `dataset.head()` to get a sense of the data's structure.

### 3. Data Cleansing

In this step, we clean and prepare the dataset for analysis. This includes:

- Removing the 'Unnamed: 0' column from the dataset.
- Checking for and reporting duplicated rows.
- Mapping and cleaning values in columns like 'WklyStudyHours', 'ParentEduc', and others to ensure data consistency.

### 4. Handling Missing Values

We check for missing values in the dataset and take appropriate actions to handle them. This includes interpolating numerical missing values ('NrSiblings') and filling categorical missing values with their mode.

### 5. Exploratory Data Analysis (EDA)

This step involves exploring the dataset to gain insights into its distribution and relationships between different features. We create various plots using Matplotlib and Seaborn to visualize the data.

### 6. Label Encoding

To prepare the data for machine learning, we map and convert categorical values to numerical values. We create mapping dictionaries for features like 'Gender', 'LunchType', 'IsFirstChild', 'TestPrep', and more, and then map the values in the dataset according to these mappings.

### 7. One Hot Encoding

We convert categorical variables in the dataset into one-hot encoded columns. This is done specifically for the 'ParentMaritalStatus' column.

### 8. Correlation Matrix

We compute the correlation matrix for the dataset and plot a heatmap of the correlations between different variables. This helps us understand the relationships between features.

### 9. Remove Unnecessary Features

To improve model performance, we filter out columns with correlations within a specific threshold range for all three target variables (MathScore, ReadingScore, and WritingScore) and remove them from the dataset.

### 10. Split the Dataset

We split the dataset into features and targets, and then save a list of one-hot encoded columns and a scaler using the pickle library.

### 11. Feature Standardization

We standardize the feature data using StandardScaler to ensure that all features have the same scale.

### 12. Split the Dataset into Training and Testing Sets

We split the dataset into training and testing sets for each target variable to prepare for model training and evaluation.

### 13. Model Evaluation

In this step, we perform hyperparameter tuning for Elastic Net Regression and Support Vector Regression (SVR) for each target variable. We use k-fold cross-validation to evaluate different hyperparameter values and plot the results to choose the best models.

### 14. Create Prediction Model

We create Elastic Net models for each target variable and fit them to the training data.

### 15. Testing New Data

To make predictions on new data, we prepare the data, one-hot encode it, scale it, and use the trained Elastic Net models to make predictions for the three target variables.

### 16. Saving Models into Pickle

Finally, we save the trained Elastic Net models into pickle files for later use.

## Accessing the Deployed Project

The project is deployed and accessible through the web application hosted at [https://exam-score-prediction.streamlit.app/](https://exam-score-prediction.streamlit.app/). You can use this web interface to interact with the project and make predictions based on the trained models.

To access the deployed project:

1. Click on the provided link: [https://exam-score-prediction.streamlit.app/](https://exam-score-prediction.streamlit.app/)
2. Use the web interface to input data and get predictions for Math, Reading, and Writing scores.

## Conclusion

This project demonstrates a comprehensive workflow for data analysis and machine learning, from data preprocessing to model evaluation and prediction. It showcases various techniques and tools commonly used in the field of data science.

If you have any questions or feedback, please feel free to contact us.

Happy analyzing and modeling!
