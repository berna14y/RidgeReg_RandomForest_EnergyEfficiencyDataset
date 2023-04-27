# RidgeReg_RandomForest_EnergyEfficiencyDataset
Ridge Regression and Random Forest Regression models are build  predictive models on the estimation of energy performance of residential buildings.
The link of dataset : http://archive.ics.uci.edu/ml/datasets/energy+efficiency.
The dataset comprises 768samples and 8 features. The features are relative compactness, surface area, wall area, roof area,overall height, orientation, glazing area, glazing area distribution. 
The two output variables are heating load (HL) and cooling load (CL), of residential buildings.

Ridge Regression and Random Forest are used to predict heading and cooling loads for the aferomentioned features of a building. 



### Ridge Regression 

Ridge regression is a linear regression technique that is used to deal with multicollinearity in the data. Multicollinearity is a phenomenon that occurs when the predictor variables in a regression model are highly correlated with each other, which can lead to unstable and unreliable estimates of the regression coefficients.

Ridge regression adds a penalty term to the least squares estimation of the regression coefficients. The penalty term is proportional to the sum of the squares of the coefficients, and a tuning parameter called the regularization parameter controls the amount of shrinkage applied to the coefficients.

The effect of the penalty term is to constrain the size of the regression coefficients, which can help to reduce their variance and improve the stability of the estimates.



For different alpha parameters, Mean Absolute Error and Mean Squared Error

    Alpha: 0.001, MSE for Y1: 9.1562, MAE for Y1: 2.1828
    Alpha: 0.01, MSE for Y1: 9.1862, MAE for Y1: 2.1901
    Alpha: 0.1, MSE for Y1: 9.4154, MAE for Y1: 2.2474
    Alpha: 1.0, MSE for Y1: 9.6535, MAE for Y1: 2.3136
    Alpha: 10.0, MSE for Y1: 10.8190, MAE for Y1: 2.4438
    
    Alpha: 0.001, MSE for Y2: 9.8932, MAE for Y2: 2.1948
    Alpha: 0.01, MSE for Y2: 9.9003, MAE for Y2: 2.1925
    Alpha: 0.1, MSE for Y2: 10.0780, MAE for Y2: 2.2242
    Alpha: 1.0, MSE for Y2: 10.3453, MAE for Y2: 2.2924
    Alpha: 10.0, MSE for Y2: 11.1643, MAE for Y2: 2.4535


After using grid search with 10-fold cross validation with 10 repetetion, the optimal values of Y1 and Y2

    Optimal alpha for Y1 (Heating Load): 0.001
    Optimal alpha for Y2 (Cooling Load): 0.001



Using optimal paramater, we get

    Y1 (Heating Load):
    MAE: 2.1119 +/- 0.2397
    MSE: 8.7708 +/- 1.7933
    
    Y2 (Cooling Load):
    MAE: 2.3284 +/- 0.2515
    MSE: 10.7225 +/- 2.4964




### Random Forest

Random forest is a popular machine learning algorithm that is used for both classification and regression tasks. It is an ensemble method that combines multiple decision trees to make predictions.

In a random forest, a large number of decision trees are constructed using different subsets of the training data and a random subset of the predictor variables at each split. The trees are constructed independently of each other

During prediction, the random forest algorithm aggregates the predictions from all the decision trees to make a final prediction.In regression tasks, the final prediction is the mean or median of the predictions from the individual trees.

Random forests have several advantages over individual decision trees, including improved accuracy, reduced overfitting, and better generalization to new data. The use of multiple trees in a random forest enable us smooth the noise and variability in the data, which can improve the robustness and accuracy of the predictions.


 After using grid search with 10-fold cross validation with 10 repetetion, the optimal values of Y1 and Y2

    Optimal parameters for Y1 (Heating Load):
    {'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    Optimal parameters for Y2 (Cooling Load):
    {'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250}



Using optimal paramater, we get

    Y1 (Heating Load):
    MAE: 0.3216 +/- 0.0472
    MSE: 0.2380 +/- 0.0874
    
    Y2 (Cooling Load):
    MAE: 0.9887 +/- 0.1531
    MSE: 2.7092 +/- 0.7935



### COMPARISON

For Y1, the Random Forest model has an MAE of 0.3216 +/- 0.0472 and an MSE of 0.2380 +/- 0.0874, compared to the Ridge Regression model's MAE of 2.1119 +/- 0.2397 and MSE of 8.7708 +/- 1.7933.

For Y2, the Random Forest model has an MAE of 0.9887 +/- 0.1531 and an MSE of 2.7092 +/- 0.7935, compared to the Ridge Regression model's MAE of 2.3284 +/- 0.2515 and MSE of 10.7225 +/- 2.4964.

To sum up, the Random Forest model appears to be a much better model for predicting both Y1 and Y2, as it has significantly lower MAE and MSE values than the Ridge Regression model.

