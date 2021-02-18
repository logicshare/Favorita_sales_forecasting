import os
import sys
import pandas as pd
from conf.base.base_conf import ROOT_DIR

items = pd.read_csv(f"{ROOT_DIR}\\Data\\d01_raw\\items.csv")
holidays = pd.read_csv(f"{ROOT_DIR}\\Data\\d01_raw\\holidays_events.csv")
oil = pd.read_csv(f"{ROOT_DIR}\\Data\\d01_raw\\oil.csv")
stores = pd.read_csv(f"{ROOT_DIR}\\Data\\d01_raw\\stores.csv")
transactions = pd.read_csv(f"{ROOT_DIR}\\Data\\d01_raw\\transactions.csv")
df = pd.read_csv(f"{ROOT_DIR}\\Data\\d01_raw\\train.csv",nrows=20000000)