import numpy as np
import pandas as pd
import os
import argparse
import joblib
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error,explained_variance_score

from conf.base.base_conf import TRAINING_DIR, MODEL_STORE, MODEL_OUTPUT
from src.d00_utils.select_model import models


def run_exps(item_family):
    
    # read in train data with folds
    df = pd.read_csv(f"{TRAINING_DIR}\\{item_family}.csv")

    X = df.drop('unit_sales', axis=1)
    y = df['unit_sales']

    # Split data for training and validation
    X_train,X_valid,y_train,y_valid = train_test_split(X, y, test_size=0.3, random_state=101)
    
    # list of cv_results dataframes
    dfs = []
    
    # Dataframe will be used later to plot actual and pred unit sales for each store by the best performing model
    actual_and_pred_df = X_valid[['date','store_nbr','item_nbr']]
    actual_and_pred_df.insert(loc=len(actual_and_pred_df.columns), column='actual_unit_sales', value=y_valid.values)
    
    # Performance metrics
    scores = ['neg_root_mean_squared_error']
    
    # write model metrics summary to a txt file
    f = open(f"{MODEL_OUTPUT}\\{item_family}_model_metrics.txt", "a")
    f.write(f"\n{str(dt.datetime.now())}\n")
    
    for model_name, model in models:
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=102)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scores)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        
        this_df = pd.DataFrame(cv_results)
        this_df['model_name'] = model_name
        
        actual_and_pred_df.insert(loc=len(actual_and_pred_df.columns), column=f'pred_unit_sales_{model_name}', value=y_pred.round())
        
        # calculate RMSE, SI and variance score
        rmse = np.sqrt(mean_squared_error(y_valid,y_pred))
        si = rmse/np.mean(y_valid)
        variance_score = explained_variance_score(y_valid,y_pred)
        
        # Feature importance
        if model_name not in ['LinearRegression','SVM']:     
            features = list(X_train.columns)
            importances = clf.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(20,12))
            plt.rc('ytick', labelsize=10)
            plt.title(f"Feature Importances for {model_name}")
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.savefig(f"{MODEL_OUTPUT}\\{item_family}_feature_importance_{model_name}.png",dpi=300)  
            plt.close()
        
        print(f"Model={model_name}, RMSE={rmse}, SI={si}, Variance Score={variance_score}")
        f.write(f"Model={model_name}, RMSE={rmse}, SI={si}, Variance Score={variance_score}\n")
        
        # save the model
        joblib.dump(clf,f"{MODEL_STORE}\\{item_family}_{model_name}.bin")
        
        dfs.append(this_df)
        
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(f"{MODEL_OUTPUT}\\{item_family}_results.csv", index=False)
    
    actual_and_pred_df.to_csv(f"{MODEL_OUTPUT}\\{item_family}_actual_and_predicted.csv", index=False)
    f.close()


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # define arguments and their types
    parser.add_argument('--item_family', type=str)
    
    # read arguments from command line
    args = parser.parse_args()

    # run the training and scoring for the item_family, fold and model specified by command line arguments
    run_exps(item_family=args.item_family)