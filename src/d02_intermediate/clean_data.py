import numpy as np
import pandas as pd
from datetime import datetime
from conf.base.base_conf import ROOT_DIR
from src.d01_data.load_data import items,holidays,oil,stores,transactions,df

# Data Cleaning

## ======================= Oil =======================

### Dates in Oil data are not continuous. Impute missing dates
oil['date'] = oil['date'].map(lambda x: datetime.strptime(x,'%Y-%m-%d'))
date_idx = pd.date_range(oil.date.min(),oil.date.max())
oil = oil.set_index('date').reindex(date_idx).rename_axis('date').reset_index()

oil.fillna(method='ffill', inplace = True) # forward fill the missing values with the lastest valid value until another valid value is found.
oil.fillna(method='bfill', inplace = True) # backward fill the missing values with the lastest valid value until another valid value is found

### Check for missing values
if oil.isna().sum().sum() != 0:
    print("Missing values found in oil data.")
else:
    print("No missing values found in oil data.")
    oil.to_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\oil_cleaned.csv", index=False)


## ======================= Train data =======================

### Check for missing values
while df.isna().sum().sum() != 0:
    print("Missing values found in train data.")
    '''
        If majority of the values (> 80 % of total records ) are missing the feature can be dropped.
        All records in onpromotion column are missing values.
    '''
    df_columns_to_drop = [items[0] for items in df.isna().sum().iteritems() if items[1] >= 0.8*len(df)]
    if len(df_columns_to_drop)>0:
        df.drop(df_columns_to_drop, axis=1, inplace=True)
        print("Columns with missing values dropped.")
    else:
        pass

print("No missing values found in train data.")
df.to_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\train_cleaned.csv", index=False)



## ======================= Items =======================

### Check for missing values
if items.isna().sum().sum() != 0:
    print("Missing values found in items data.")   
else:
    print("No missing values found in items data.")
    items.to_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\items_cleaned.csv", index=False)


    
## ======================= Holidays =======================
### Check for missing values
if holidays.isna().sum().sum() != 0:
    print("Missing values found in holidays data.")
else:
    print("No missing values found in holidays data.")
    holidays.to_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\holidays_cleaned.csv", index=False)

    

## ======================= Stores =======================
### Check for missing values
if stores.isna().sum().sum() != 0:
    print("Missing values found in stores data.")
else:
    print("No missing values found in stores data.")
    stores.to_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\stores_cleaned.csv", index=False)

    

## ======================= Transactions =======================
### Check for missing values
if transactions.isna().sum().sum() != 0:
    print("Missing values found in transactions data.")
else:
    print("No missing values found in transactions data.")
    transactions.to_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\transactions_cleaned.csv", index=False)
