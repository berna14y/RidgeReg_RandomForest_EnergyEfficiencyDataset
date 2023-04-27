# RidgeReg_RandomForest_EnergyEfficiencyDataset
Ridge Regression and Random Forest Regression models are build  predictive models on the estimation of energy performance of residential buildings.
The link of dataset : http://archive.ics.uci.edu/ml/datasets/energy+efficiency.
The dataset comprises 768samples and 8 features. The features are relative compactness, surface area, wall area, roof area,overall height, orientation, glazing area, glazing area distribution. 
The two output variables are heating load (HL) and cooling load (CL), of residential buildings.

Ridge Regression and Random Forest are used to predict heading and cooling loads for the aferomentioned features of a building. 


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
```

### Ridge Regression 

Ridge regression is a linear regression technique that is used to deal with multicollinearity in the data. Multicollinearity is a phenomenon that occurs when the predictor variables in a regression model are highly correlated with each other, which can lead to unstable and unreliable estimates of the regression coefficients.

Ridge regression adds a penalty term to the least squares estimation of the regression coefficients. The penalty term is proportional to the sum of the squares of the coefficients, and a tuning parameter called the regularization parameter controls the amount of shrinkage applied to the coefficients.

The effect of the penalty term is to constrain the size of the regression coefficients, which can help to reduce their variance and improve the stability of the estimates.



```python
import numpy as np
import pandas as pd

# Load the data into a Pandas DataFrame
data = pd.read_excel('ENB2012_data.xlsx')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>Y1</th>
      <th>Y2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.90</td>
      <td>563.5</td>
      <td>318.5</td>
      <td>122.50</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>20.84</td>
      <td>28.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    data.iloc[:, :-2],  # Features
    data.iloc[:, -2],   # Target variable 1 (heating load)
    data.iloc[:, -1],   # Target variable 2 (cooling load)
    test_size=0.2,      # 20% of data as test set
    random_state=42)    # Fix random seed for reproducibility
```


```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]  # Different alpha values to test

# Train and test the Ridge Regression model for heating load (y1)
for alpha in alphas:
    model1 = Ridge(alpha=alpha)
    model1.fit(X_train, y1_train)
    y1_pred = model1.predict(X_test)
    mse = mean_squared_error(y1_test, y1_pred)
    mae = mean_absolute_error(y1_test, y1_pred)
    print(f"Alpha: {alpha}, MSE for Y1: {mse:.4f}, MAE for Y1: {mae:.4f}")
    
print()    
# Train and test the Ridge Regression model for cooling load (y2)
for alpha in alphas:
    model2 = Ridge(alpha=alpha)
    model2.fit(X_train, y2_train)
    y2_pred = model2.predict(X_test)
    mse = mean_squared_error(y2_test, y2_pred)
    mae = mean_absolute_error(y2_test, y2_pred)
    print(f"Alpha: {alpha}, MSE for Y2: {mse:.4f}, MAE for Y2: {mae:.4f}")


```

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



```python

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, RepeatedKFold


# The cross_validate_ridge function uses repeated k-fold cross-validation with 10 folds and 10 repeats 
# to evaluate the performance of the ridge regression model

def cross_validate_ridge(X, y, alpha):
    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)  # 10-fold and 10 repeats cross-validation
    mse_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
    
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    return mean_mae, std_mae, mean_mse, std_mse

```


```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Define the Ridge Regression model
ridge = Ridge()

# Define the range of alpha values to test
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

# Define the grid search parameters
param_grid = {'alpha': alphas}

# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(ridge, param_grid=param_grid, cv=10, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y1_train)

# Print the optimal alpha for Y1
print(f"Optimal alpha for Y1 (Heating Load): {grid_search.best_params_['alpha']}")

# Rebuild the model with the optimal alpha for Y1
ridge_y1 = Ridge(alpha=grid_search.best_params_['alpha'])
ridge_y1.fit(X_train, y1_train)

# Perform grid search for Y2
grid_search.fit(X_train, y2_train)

# Print the optimal alpha for Y2
print(f"Optimal alpha for Y2 (Cooling Load): {grid_search.best_params_['alpha']}")

# Rebuild the model with the optimal alpha for Y2
ridge_y2 = Ridge(alpha=grid_search.best_params_['alpha'])
ridge_y2.fit(X_train, y2_train)

```

    Optimal alpha for Y1 (Heating Load): 0.001
    Optimal alpha for Y2 (Cooling Load): 0.001





    Ridge(alpha=0.001)




```python
alpha = 00.1  # Optimal alpha 

# Perform 10-fold and 10 repeat cross-validation for Y1 (heating load)
mean_mae1, std_mae1, mean_mse1, std_mse1 = cross_validate_ridge(X_train, y1_train, alpha)

print(f"Y1 (Heating Load):")
print(f"MAE: {mean_mae1:.4f} +/- {std_mae1:.4f}")
print(f"MSE: {mean_mse1:.4f} +/- {std_mse1:.4f}\n")

# Perform 10-fold and 10 repeat cross-validation for Y2 (cooling load)

mean_mae2, std_mae2, mean_mse2, std_mse2 = cross_validate_ridge(X_train, y2_train, alpha)

print(f"Y2 (Cooling Load):")
print(f"MAE: {mean_mae2:.4f} +/- {std_mae2:.4f}")
print(f"MSE: {mean_mse2:.4f} +/- {std_mse2:.4f}")

```

    Y1 (Heating Load):
    MAE: 2.1119 +/- 0.2397
    MSE: 8.7708 +/- 1.7933
    
    Y2 (Cooling Load):
    MAE: 2.3284 +/- 0.2515
    MSE: 10.7225 +/- 2.4964


### Another way


##### Split into input (X) and output (y) variables
X = data.iloc[:, :-2].values
y = data.iloc[:, -2:].values
##### Define the alpha parameters to test
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

##### Find the optimal alpha parameter using RidgeCV
ridge_cv = RidgeCV(alphas=alphas, cv=10).fit(X_train, y_train)
alpha_opt = ridge_cv.alpha_
print("Optimal alpha parameter:", alpha_opt)
##### Train the Ridge model using the optimal alpha parameter
ridge = Ridge(alpha=alpha_opt).fit(X_train, y_train)
##### perform cross-validation with 10-fold 10-repetition strategy for Y1
ridge_y1 = cross_validate(ridge, X, y1, cv=10, n_jobs=-1, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error'), return_train_score=True)

##### perform cross-validation with 10-fold 10-repetition strategy for Y2
ridge_y2 = cross_validate(ridge, X, y2, cv=10, n_jobs=-1, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error'), return_train_score=True)

##### calculate the mean and standard deviation of the MAE and MSE scores for both Y1 and Y2
mae_ridge_y1 = -ridge_y1['test_neg_mean_absolute_error']

mse_ridge_y1 = -ridge_y1['test_neg_mean_squared_error']

mae_ridge_y2 = -ridge_y2['test_neg_mean_absolute_error']

mse_ridge_y2 = -ridge_y2['test_neg_mean_squared_error']

print('RIDGE REGRESSION')

print('Y1 MAE Mean:', mean(mae_ridge_y1), 'Std:', std(mae_ridge_y1))

print('Y1 MSE Mean:', mean(mse_ridge_y1), 'Std:', std(mse_ridge_y1))

print('Y2 MAE Mean:', mean(mae_ridge_y2), 'Std:', std(mae_ridge_y2))

print('Y2 MSE Mean:', mean(mse_ridge_y2), 'Std:', std(mse_ridge_y2))

### Random Forest

Random forest is a popular machine learning algorithm that is used for both classification and regression tasks. It is an ensemble method that combines multiple decision trees to make predictions.

In a random forest, a large number of decision trees are constructed using different subsets of the training data and a random subset of the predictor variables at each split. The trees are constructed independently of each other

During prediction, the random forest algorithm aggregates the predictions from all the decision trees to make a final prediction.In regression tasks, the final prediction is the mean or median of the predictions from the individual trees.

Random forests have several advantages over individual decision trees, including improved accuracy, reduced overfitting, and better generalization to new data. The use of multiple trees in a random forest enable us smooth the noise and variability in the data, which can improve the robustness and accuracy of the predictions.


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the RandomForestRegressor model
rf = RandomForestRegressor(random_state=42)

# Define the grid search parameters
param_grid = {'n_estimators': [10, 50, 100, 250, 500],
              'max_depth': [50, 150, 250],
              'min_samples_split': [2, 3],
              'min_samples_leaf': [1, 2, 3]}

# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y1_train)

# Print the optimal parameters for Y1
print("Optimal parameters for Y1 (Heating Load):")
print(grid_search.best_params_)

# Rebuild the model with the optimal parameters for Y1
rf_y1 = RandomForestRegressor(random_state=42, **grid_search.best_params_)
rf_y1.fit(X_train, y1_train)

# Perform grid search for Y2
grid_search.fit(X_train, y2_train)

# Print the optimal parameters for Y2
print("Optimal parameters for Y2 (Cooling Load):")
print(grid_search.best_params_)

# Rebuild the model with the optimal parameters for Y2
rf_y2 = RandomForestRegressor(random_state=42, **grid_search.best_params_)
rf_y2.fit(X_train, y2_train)
```

    Optimal parameters for Y1 (Heating Load):
    {'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    Optimal parameters for Y2 (Cooling Load):
    {'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250}





    RandomForestRegressor(max_depth=50, n_estimators=250, random_state=42)




```python

# The cross_validate_ridge function uses repeated k-fold cross-validation with 10 folds and 10 repeats 
# to evaluate the performance of the random forest model

def cross_validate_RandomForest(X, y, max_depth,min_samples_leaf,min_samples_split,n_estimators):
    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)   # 10-fold cross-validation with random shuffle
    mae_scores = []
    mse_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestRegressor(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
    
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    return mean_mae, std_mae, mean_mse, std_mse
```


```python

# Perform 10-fold cross-validation for Y1 (heating load)
mean_mae1, std_mae1, mean_mse1, std_mse1 = cross_validate_RandomForest(X_train, y1_train, 50,1,2,250)

print(f"Y1 (Heating Load):")
print(f"MAE: {mean_mae1:.4f} +/- {std_mae1:.4f}")
print(f"MSE: {mean_mse1:.4f} +/- {std_mse1:.4f}\n")

# Perform 10-fold cross-validation for Y2 (cooling load)

mean_mae2, std_mae2, mean_mse2, std_mse2 = cross_validate_RandomForest(X_train, y2_train, 50,1,2,250)

print(f"Y2 (Cooling Load):")
print(f"MAE: {mean_mae2:.4f} +/- {std_mae2:.4f}")
print(f"MSE: {mean_mse2:.4f} +/- {std_mse2:.4f}")
```

    Y1 (Heating Load):
    MAE: 0.3216 +/- 0.0472
    MSE: 0.2380 +/- 0.0874
    
    Y2 (Cooling Load):
    MAE: 0.9887 +/- 0.1531
    MSE: 2.7092 +/- 0.7935


### Another way
##### define the RandomForestRegressor model with the optimal hyperparameters
rf_model = RandomForestRegressor(n_estimators=100, max_depth=50, min_samples_split=2, min_samples_leaf=3, random_state=42)

##### perform cross-validation with 10-fold 10-repetition strategy for Y1
rf_y1 = cross_validate(rf_model, X, y1, cv=10, n_jobs=-1, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error'), return_train_score=True)

##### perform cross-validation with 10-fold 10-repetition strategy for Y2
rf_y2 = cross_validate(rf_model, X, y2, cv=10, n_jobs=-1, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error'), return_train_score=True)

##### calculate the mean and standard deviation of the MAE and MSE scores for both Y1 and Y2
mae_rf_y1 = -rf_y1['test_neg_mean_absolute_error']

mse_rf_y1 = -rf_y1['test_neg_mean_squared_error']

mae_rf_y2 = -rf_y2['test_neg_mean_absolute_error']

mse_rf_y2 = -rf_y2['test_neg_mean_squared_error']

print('RANDOM FOREST')

print('Y1 MAE Mean:', mean(mae_rf_y1), 'Std:', std(mae_rf_y1))

print('Y1 MSE Mean:', mean(mse_rf_y1), 'Std:', std(mse_rf_y1))

print('Y2 MAE Mean:', mean(mae_rf_y2), 'Std:', std(mae_rf_y2))

print('Y2 MSE Mean:', mean(mse_rf_y2), 'Std:', std(mse_rf_y2))

### COMPARISON

For Y1, the Random Forest model has an MAE of 0.3216 +/- 0.0472 and an MSE of 0.2380 +/- 0.0874, compared to the Ridge Regression model's MAE of 2.1119 +/- 0.2397 and MSE of 8.7708 +/- 1.7933.

For Y2, the Random Forest model has an MAE of 0.9887 +/- 0.1531 and an MSE of 2.7092 +/- 0.7935, compared to the Ridge Regression model's MAE of 2.3284 +/- 0.2515 and MSE of 10.7225 +/- 2.4964.

To sum up, the Random Forest model appears to be a much better model for predicting both Y1 and Y2, as it has significantly lower MAE and MSE values than the Ridge Regression model.


```python

```
