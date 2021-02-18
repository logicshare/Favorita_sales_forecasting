import pandas as pd
from conf.base.base_conf import TRAINING_DIR

def get_training_data(item_family):
    item_family = item_family.replace(' ','_').replace(',','_').replace('/','_')
    return pd.read_csv(f"{TRAINING_DIR}\\{item_family}.csv")
