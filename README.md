# credit-score-predictor
Using the sklearn Random Forest Library, I have created a model that predicts whether a users Credit Score is 'Poor', 'Standard' or 'Good' using a dataset that I found on Kaggle.

## Pre-processing
This dataset was very messy, so a lot of cleaning needed to be done!

I realised that I only needed to read in the training dataset at first, and then just apply all those same changes to the test set when I needed to test the model.

To gain an overview of the dataset, I used 4 common commands:
1. `head` to visualise the first few rows of the dataset
2. `shape` to see how many rows and columns there are in the dataframe
3. `describe` to see the key statistics of each feature
4. `info` to see the data types of features and other details

I also used `.isna().sum()` to count the number of NaN rows. For rows that had loads of NaN entries, I simply removed these from my analysis as they weren't of any use.

Below are other rows that I removed along with reasoning:
1. `ID`, `Customer_ID`, `Month`, `Name`, `SSN`, `Type_of_Loan`: removed not due to data issues, but because the features are not good predictors of someone's credit score
2. `Monthly_Inhand_Salary`: 15002 NaN rows - `Annual_Income` is good enough

From a first glance, key issues that I noticed with entries in the CSV included underscores in numerical data which was easily fixed using `str.replace()`. Also, rows with negative values were removed as this is not possible, as well as duplicate rows.
Another issue was pandas reading in the columns as incorrect data types, so I had to manually define each `dtype`.

Some other basic amendments that I made to features:
1. `Num_of_Delayed_Payment`: 7002 NaN rows, so I interpreted these as 0 delayed payments
2. `Num_Credit_Inquiries`: same as above


### Converting non-numerical data to labels
An issue that I ran into using this library with this specific dataset is that non-numerical columns are not directly supported. Following some StackOverflow searching, I found that the best solution to this problem is converting such columns to labels using sklearn's Label Encoder.

## The model

## Testing

## Improvements
