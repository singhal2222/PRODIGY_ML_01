House Price Prediction using Linear Regression
This project is part of the Prodigy InfoTech internship in the Machine Learning domain. The task involves implementing a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

Project Overview
The goal of this project is to build a predictive model that estimates the price of houses. The dataset used for this task contains information about various houses, including their square footage, number of bedrooms, and number of bathrooms. The model will use these features to predict house prices.

Dataset
The dataset used in this project is sourced from Kaggle's House Prices - Advanced Regression Techniques competition. The data includes the following key features:

SquareFootage
NumberOfBedrooms
NumberOfBathrooms
You can download the dataset from Kaggle.

Project Structure
The project consists of the following files:

train.csv: Training dataset with house features and prices.
test.csv: Test dataset with house features.
house_price_prediction.ipynb: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
Implementation
Data Preprocessing
Loading the Data: The training and test datasets are loaded into pandas DataFrames.
Handling Missing Values: Missing values in the features are handled by filling them with the mean value of the respective columns.
Feature Scaling: The features are scaled using StandardScaler to ensure they contribute equally to the model.
Model Training
Three different regression models are trained and evaluated:

Linear Regression
Ridge Regression
Lasso Regression
The models are trained using the training dataset, and their performance is evaluated using Mean Squared Error (MSE).

Model Evaluation
The performance of each model is evaluated using the Mean Squared Error (MSE) metric. The model with the lowest MSE is selected as the best model.

Results
Linear Regression Mean Squared Error: 1576282550.6757114
Ridge Regression Mean Squared Error: 1575585346.935686
Lasso Regression Mean Squared Error: 1576289846.5532968
Visualization
The clusters are visualized using a scatter plot, with different colors representing different clusters. This helps in understanding the distribution of customers based on their annual income and spending score.

Conclusion
This project successfully demonstrates the use of linear regression and its variants to predict house prices. The Ridge Regression model performed the best in this scenario. This task provided valuable insights into data preprocessing, model training, and evaluation in the context of real estate price prediction.

How to Run
Clone this repository.
Download the dataset from Kaggle and place train.csv and test.csv in the project directory.
Open house_price_prediction.ipynb in Jupyter Notebook or Google Colab.
Run all the cells to see the results.
Future Work
Future improvements can include:

Adding more features to the model for better accuracy.
Exploring other regression algorithms.
Performing hyperparameter tuning for the models.
Acknowledgements
Kaggle for providing the dataset.
Prodigy InfoTech for the internship opportunity.
