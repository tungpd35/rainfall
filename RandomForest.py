import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_score as score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

#Tiền sử lý dữ liệu
data9.dropna(inplace=True)
data10.dropna(inplace=True)

X = pd.concat([data9[['B09B','B10B','B12B','B14B','B16B','I2B','IRB','WVB','CAPE','TCC','TCW','TCWV']],data10[['B09B','B10B','B12B','B14B','B16B','I2B','IRB','WVB','CAPE','TCC','TCW','TCWV']]],axis=0) 
y = pd.concat([data9['value'],data10['value']],axis=0) 

#Chia dataset train,validation,testing 70/20/10
X_train, X_val, y_train, y_val = split(X,y,test_size=0.2,random_state=8) 
X_train, X_test, y_train, y_test = split(X_train, y_train, test_size=0.125, random_state=8)
rf = RandomForestRegressor()

# Tìm kiếm các tham số tối ưu cho mô hình
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# In ra các tham số tối ưu cho mô hình
print(grid_search.best_params_)
# Đánh giá mô hình trên tập validate 
y_pred = grid_search.predict(X_val)
score = grid_search.score(X_val, y_val)
print('Đánh giá mô hình trên tập validate')
print("Accuracy: ",score)
print('RMSE:', np.sqrt(mse(y_pred, y_val)))
print('MAE: ', mae(y_pred, y_val))
print('')
# Đánh giá mô hình trên dữ liệu kiểm tra
y_pred = grid_search.predict(X_test)
score = grid_search.score(X_test, y_test)
print('Đánh giá mô hình trên dữ liệu testing')
print("Accuracy: ", score)
print('RMSE:', np.sqrt(mse(y_pred, y_test)))
print('MAE: ', mae(y_pred, y_test))
