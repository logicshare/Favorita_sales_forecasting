from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
    
models = [
          ("LinearRegression",LinearRegression()),
          ("Decision_Tree",DecisionTreeRegressor(random_state=220)),
          ("SVM",SVR()),
          ("RandomForest",RandomForestRegressor(random_state=420)),
          ("XGBoost",XGBRegressor(random_state=520))
         ]
