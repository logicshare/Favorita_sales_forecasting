import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from zipfile import ZipFile
import os

from conf.base.base_conf import MODEL_OUTPUT, REPORTING_DIR

def plot_actual_predicted_unit_sales(item_family):
    
    actual_and_pred_df = pd.read_csv(f"{MODEL_OUTPUT}\\{item_family}_actual_and_predicted.csv")
    actual_and_pred_df['date'] = actual_and_pred_df['date'].map(datetime.fromordinal)
    actual_and_pred_agg_df = actual_and_pred_df[['date','store_nbr','actual_unit_sales','pred_unit_sales_XGBoost']].groupby(['date','store_nbr']).sum()
    actual_and_pred_agg_df.reset_index(inplace=True)
    

    for str_nbr in actual_and_pred_agg_df['store_nbr'].unique():    
        plt.figure(figsize=(20,12))
        sns.set(font_scale=1.5)
        sns.lineplot(x='date', y='actual_unit_sales', data= actual_and_pred_agg_df.loc[actual_and_pred_agg_df['store_nbr'] == str_nbr], legend='brief', label='Actual unit sales')
        # XGBoost is the best performing model
        sns.lineplot(x='date', y='pred_unit_sales_XGBoost', data= actual_and_pred_agg_df.loc[actual_and_pred_agg_df['store_nbr'] == str_nbr,], legend='brief', label='Predicted unit sales')
        plt.ylabel('Date')
        plt.ylabel('Unit Sales Quantity')
        plt.xticks(rotation=90)
        plt.title(f'Actual vs Predicted unit sales in {item_family} for store {str_nbr}')
        plt.savefig(f"{REPORTING_DIR}\\{item_family}_actual_vs_predicted_unit_sales_for_store_{str_nbr}.png",dpi=300)
        plt.close()
        
    # add generated plots to zip file
    zipobj = ZipFile(f"{REPORTING_DIR}\\{item_family}_actual_vs_predicted_unit_sales_plots.zip",'w')
    
    for str_nbr in actual_and_pred_agg_df['store_nbr'].unique():
        file_path = f"{REPORTING_DIR}\\{item_family}_actual_vs_predicted_unit_sales_for_store_{str_nbr}.png"
        zipobj.write(file_path,os.path.basename(file_path))
        # delete the .png file
        try:
            os.remove(file_path)
        except OSError as e:
            print("Error: %s : %s" % (file_path, e.strerror))
            
    zipobj.close()

        

    
if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # define arguments and their types
    parser.add_argument('--item_family', type=str)
    
    # read arguments from command line
    args = parser.parse_args()

    # run the training and scoring for the item_family, fold and model specified by command line arguments
    plot_actual_predicted_unit_sales(item_family=args.item_family)