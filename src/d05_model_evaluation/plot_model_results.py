import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from conf.base.base_conf import MODEL_OUTPUT

def plot_model_metrics(item_family):
    
    performance_results = pd.read_csv(f"{MODEL_OUTPUT}\\{item_family}_results_performance.csv")
    time_results = pd.read_csv(f"{MODEL_OUTPUT}\\{item_family}_results_time.csv")

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=1.5)
    sns.boxplot(x="model_name", y="values", hue="metrics", data=performance_results, palette="Set3")
    plt.legend(loc=1, bbox_to_anchor=(1.05,1), borderaxespad=0., fontsize='xx-small')
    plt.title('Comparison of models by Performance Metric')
    plt.savefig(f"{MODEL_OUTPUT}\\{item_family}_compare_models_by_performance.png",dpi=300)
    plt.close()

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=1.5)
    sns.boxplot(x="model_name", y="values", hue="metrics", data=time_results, palette="Set3")
    plt.legend(loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0., fontsize='xx-small')
    plt.title('Comparison of models by Time Metric')
    plt.savefig(f"{MODEL_OUTPUT}\\{item_family}_compare_models_by_time.png",dpi=300)
    plt.close()

 
    
    
if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # define arguments and their types
    parser.add_argument('--item_family', type=str)
    
    # read arguments from command line
    args = parser.parse_args()

    # run the training and scoring for the item_family, fold and model specified by command line arguments
    plot_model_metrics(item_family=args.item_family)
