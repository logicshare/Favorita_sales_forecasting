import os
import argparse
import pandas as pd
import numpy as np

from conf.base.base_conf import MODEL_OUTPUT
from src.d00_utils.select_model import models

def compare_models(item_family):
    
    # read model results
    results = pd.read_csv(f"{MODEL_OUTPUT}\\{item_family}_results.csv")
    
    bootstraps = []
    for model_name,model in models:
        model_result_df = results.loc[results.model_name == model_name]
        this_bootstrap = model_result_df.sample(n=30, replace=True)
        bootstraps.append(this_bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    
    results_long = pd.melt(bootstrap_df, id_vars=['model_name'], var_name='metrics', value_name='values')
    
    ## Performance metrics
    # get df without fit_time and score_time data
    perf_results_long = results_long.loc[~results_long['metrics'].isin(['fit_time','score_time'])].sort_values(by='values')
    perf_results_long.to_csv(f"{MODEL_OUTPUT}\\{item_family}_results_performance.csv", index=False)
    
    ## Time metrics
    # get df with fit_time and score_time data
    time_results_long = results_long.loc[results_long['metrics'].isin(['fit_time','score_time'])].sort_values(by='values')
    time_results_long.to_csv(f"{MODEL_OUTPUT}\\{item_family}_results_time.csv", index=False)
    

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # define arguments and their types
    parser.add_argument('--item_family', type=str)
    
    # read arguments from command line
    args = parser.parse_args()

    # run the training and scoring for the item_family, fold and model specified by command line arguments
    compare_models(item_family=args.item_family)