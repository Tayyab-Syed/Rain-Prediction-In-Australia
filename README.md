# Model Comparison and Dashboard for Weather Data Analysis

This repository contains Python code for comparing different machine learning models for weather data analysis, as well as a dashboard built using Dash to visualize the performance metrics of these models.

## Overview

The dataset used for this analysis is named `Weather_Data.csv`, and it includes various features related to weather conditions. The goal is to predict whether it will rain tomorrow (binary classification: Yes/No) based on the given features.

## Code Structure

### 1. Data Preprocessing and Model Training

The initial part of the code involves loading the dataset, preprocessing the data, and training several machine learning models:
- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Logistic Regression

The models are evaluated using metrics such as accuracy, F1 score, Jaccard index, and log loss where applicable.

### 2. Model Comparison

The script compares the performance of the models and determines the best-performing model based on accuracy. The confusion matrix for the best-performing model is visualized using seaborn and Plotly.

### 3. Dashboard Creation with Dash

The code utilizes Dash, a Python web application framework, to create an interactive dashboard for model comparison. The dashboard includes dropdowns to select different models and displays plots for accuracy, F1 score, Jaccard index, and the confusion matrix for the best-performing model.

## Instructions to Run the Code

1. Ensure you have the required dependencies installed. You can install them using:

   ```bash
   pip install dash pandas scikit-learn seaborn matplotlib plotly
   ```

2. Download the `Weather_Data.csv` file.

3. Run the Python script to execute the analysis and create the dashboard.

   ```bash
   python script_name.py
   ```

4. Open your web browser and go to [http://127.0.0.1:8050/](http://127.0.0.1:8050/) to view the dashboard.

## Results and Visualization

The script generates various classification metrics for each model and creates visualizations, including bar plots for accuracy, F1 score, and Jaccard index. The dashboard provides an interactive way to explore and compare the models, allowing users to select different models and view corresponding performance metrics and confusion matrices.

Feel free to experiment with the code, models, and dashboard layout to enhance the analysis based on your requirements.
