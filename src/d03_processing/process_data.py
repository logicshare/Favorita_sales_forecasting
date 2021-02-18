import numpy as np
import pandas as pd
from datetime import datetime
from conf.base.base_conf import ROOT_DIR


items = pd.read_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\items_cleaned.csv")
oil = pd.read_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\oil_cleaned.csv")
df = pd.read_csv(f"{ROOT_DIR}\\Data\\d02_intermediate\\train_cleaned.csv")

# ================================== Drop ID columns from train Data =============================
# Drop ID column from train data
df.drop('id', axis=1, inplace=True)
print("ID column dropped.")

# ================================== Merge Data =============================

# Merge items to train data
print("Merging item data.")
df = df.merge(items,on='item_nbr',how='left')

# Merge oil to train data
print("Merging oil data.")
oil['date'] = oil['date'].map(lambda x: datetime.strptime(x,'%Y-%m-%d'))
df['date'] = df['date'].map(lambda x: datetime.strptime(x,'%Y-%m-%d'))
df = df.merge(oil,on='date',how='left')

print("Merge completed successfully.")

# ================================== Feature Engineering =============================

## Previous day value of unit sales for an item and store
df['previous_day_unit_sales'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr'])['unit_sales'].shift(1,fill_value=0)

## Previous 7 days rolling average value of unit sales for an item and store
df['unit_sales_7_days_rolling_mean'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr']).rolling(7,1)['unit_sales'].mean().droplevel(level=[0,1])  # level in droplevel is based on the number of columns in the group by clause
df['unit_sales_7_days_rolling_mean'] = df['unit_sales_7_days_rolling_mean'].map(lambda x: round(x,0))

## Previous 6 days rolling average value of unit sales for an item and store
df['unit_sales_6_days_rolling_mean'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr']).rolling(6,1)['unit_sales'].mean().droplevel(level=[0,1]) 
df['unit_sales_6_days_rolling_mean'] = df['unit_sales_6_days_rolling_mean'].map(lambda x: round(x,0))

## Previous 5 days rolling average value of unit sales for an item and store
df['unit_sales_5_days_rolling_mean'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr']).rolling(5,1)['unit_sales'].mean().droplevel(level=[0,1]) 
df['unit_sales_5_days_rolling_mean'] = df['unit_sales_5_days_rolling_mean'].map(lambda x: round(x,0))

## Previous 4 days rolling average value of unit sales for an item and store
df['unit_sales_4_days_rolling_mean'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr']).rolling(4,1)['unit_sales'].mean().droplevel(level=[0,1]) 
df['unit_sales_4_days_rolling_mean'] = df['unit_sales_4_days_rolling_mean'].map(lambda x: round(x,0))

## Previous 3 days rolling average value of unit sales for an item and store
df['unit_sales_3_days_rolling_mean'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr']).rolling(3,1)['unit_sales'].mean().droplevel(level=[0,1]) 
df['unit_sales_3_days_rolling_mean'] = df['unit_sales_3_days_rolling_mean'].map(lambda x: round(x,0))

## Previous 2 days rolling average value of unit sales for an item and store
df['unit_sales_2_days_rolling_mean'] = df[['store_nbr','item_nbr','unit_sales']].groupby(['store_nbr','item_nbr']).rolling(2,1)['unit_sales'].mean().droplevel(level=[0,1]) 
df['unit_sales_2_days_rolling_mean'] = df['unit_sales_2_days_rolling_mean'].map(lambda x: round(x,0))

## Day of Week
df['day_of_week'] = df['date'].map(lambda x: datetime.strftime(x,'%A'))

print("Feature Engineering completed successfully.")

# ================================== Convert Categorical Variables =============================

day_of_week_df = pd.get_dummies(df['day_of_week'], prefix='day_of_week', drop_first = True)
df.drop('day_of_week', axis=1, inplace=True)
df = pd.concat([df,day_of_week_df], axis=1)
del([day_of_week_df])

print("Categorical Variables converted successfully.")

# ================================== Convert date to ordinal format =============================

df['date'] = df['date'].map(lambda x: datetime.toordinal(x))

print("Date converted to ordinal format successfully.")

# ================================== Split train data based on item family =============================

for item_family in items['family'].unique():
    family_name = item_family.replace(' ','_').replace(',','_').replace('/','_')
    family_df = df[df['family'] == item_family]
    # drop family column
    family_df.drop('family', axis=1, inplace=True)
    family_df.to_csv(f"{ROOT_DIR}\\Data\\d03_processed\\{family_name}.csv", index=False)
    print(f"Train data created for item family {family_name}")

print(f"Train data files saved at {ROOT_DIR}\\Data\\d03_processed")
del(df)