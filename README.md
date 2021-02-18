# Favorita_sales_forecasting

### Dain Thengunnal    
### 10 Feb 2021

## Introduction

Brick-and-mortar grocery stores are always in a delicate balance with purchasing and sales forecasting.
Predict a little over, and grocers are stuck with overstocked, perishable goods.
Guess a little under, and popular items quickly sell out, leaving money on the table and customers fuming.

Corporacion Favorita is a large Ecuadorian-based grocery retailer who operates 54 supermarkets across Ecuador.
They currently rely on subjective forecasting methods with very little data to back them up and very little automation to execute plans.
They are excited to see how machine learning could better ensure they please customers by having just enough of the right products at the right time.

## Challenge

Build a model that more accurately predict sales for a given product at a given store from August 16th to August 31st 2017.

## Data source

Corporacion Favorita has provided historical sales data from 2012 to August 15th, 2017 at the following link.
https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data

## Modelling

Linear Regression, Decision Tree, Support Vector Machine, Random Forest and XGBoost algorithms were used train for each item family.

## Results

XGBoost is the winner in predicting unit sales for a given product at a given store for a given date with minimal error.

## Conclusion

Models were trained on a small sample data (20 million out of 125 million training records) due to memory limitations.
Prediction accuracy can be improved by using larger training sample size, fine tuning the model parameters and including weather and demographics data.