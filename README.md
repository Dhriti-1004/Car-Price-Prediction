# Car-Price-Prediction
Car price predictor model using machine learning
This dataset has been downloaded from Kaggle. 
It contains 4340 rows and 8 columns:
1) name: name of the car along with model 
2) year: the year it was manufactured
3) 6: the price at which the car will be resold (target column)
4) km_driven: number of kilometres driven by the car
5) fuel: type of fuel used in the car
6) seller_type: type of seller- dealer or individual
7) transmission: whether the car is manual or automatic
8) owner: the number of owners of the car 

Using pandas, this data was loaded into a dataframe and its basic information was studied. Many insights about the data were found during Exploratory Data Analysis (EDA) and the impact of all the columns on the output were studied. The data was found to be mostly linear with some aspects being non linear.
# Data preprocessing:
The data types of the data were changed as required, and the categorical columns were converted into numbers so that the data can be trained using linear regression and other such linear models.
Also, some outliers in the selling price were capped. The data was scaled using min-max scaling technique and log transformation was applied to many columns so that they become more normally distributed. 
Polynomial features were introduced into the dataset so that the non linear aspects of the data can also be captured by the model.
After train and test split, many models were trained.
# Models tried: 
1) Linear Regression: gave a cross validation score of 0.686, it was applied because a lot of relationships in the data seemed linear.
2) Lasso Regression: this regularization was applied so that the results improve by introduction of some noise but the results were find to be similar.
3) Ridge Regression: the results were similar
4) Bagging: this ensemble technique gives good results on many datasets and was applied twice using linear as well as lasso regressions as the base models, but this also gave similar results.
5) Random Forest: with decision trees as their base, these give good results on a large variety of datasets. The results improved by 2 percent rising to 0.709.
6) XGBoost: improving further on decision trees, this powerful machine learning algorithm gave the highest cross validation score of 0.716 and was chosen as the final model for this prediction.
With XGBoost as the algorithm, the final model was saved using pickle module. 
The residuals were plotted and were found to be normally distributed, indicating the choice of model was correct and the cross validation score of 0.716 is the final accuracy for this model.

