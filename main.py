import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

numerical_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]



def show_stats(dataset):

    print(dataset.shape)

    # Check how many entries are 0
    print(dataset.isna().sum())

    for col in dataset.columns:
        print(f"Column '{col}' has data type: {dataset[col].dtype}")

def formatting(dataset):

    # Remove underscores from numerical columns
    for col in numerical_columns:
        dataset[col] = dataset[col].str.replace('_', '')

    # Specify desired data types for each column
    dtypes = {'ID': object, 'Customer_ID': object, 'Month': object, 'Name': object, 'Age': int, 'SSN': object, 'Occupation': object, 'Annual_Income': float, 'Monthly_Inhand_Salary': float, 'Num_Bank_Accounts': int, 'Num_Credit_Card': int, 'Interest_Rate': int, 'Num_of_Loan': int, 'Type_of_Loan': object, 'Delay_from_due_date': int, 'Num_of_Delayed_Payment': int, 'Changed_Credit_Limit': float, 'Num_Credit_Inquiries': int, 'Credit_Mix': object, 'Outstanding_Debt': float, 'Credit_Utilization_Ratio': float, 'Credit_History_Age': int, 'Payment_of_Min_Amount': bool, 'Total_EMI_per_month': float, 'Amount_invested_monthly': float, 'Payment_Behaviour': object, 'Monthly_Balance': float, 'Credit_Score': object}

    for col, dtype in dtypes.items():
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce').astype(dtype)

    # Remove negative values
    for col in numerical_columns:
        dataset = dataset[dataset[col] >= 0]

    dataset = dataset.drop_duplicates()
    # dataset = dataset.dropna(inplace=True)
    print(dataset.describe())

    columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
    print(columns_with_nan)


    print(dataset.head())

def interpolate_missing_values(dataset):

    # INTERPOLATION
    rows_to_interpolate = ['Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Amount_invested_monthly', 'Monthly_Balance']

def convert_objects_to_labels(dataset):

    # Specify the names of the categorical columns and converting them to labels
    categorical_columns = ["Month", "Occupation", "Type_of_Loan", "Credit_Mix", "Payment_Behaviour", "Payment_of_Min_Amount", "Credit_History_Age"]

    le = preprocessing.LabelEncoder()

    for col_name in categorical_columns:
        col_index = dataset.columns.get_loc(col_name)  # Find the index of the column by column name
        dataset[col_name] = le.fit_transform(dataset[col_name])


def remove_underscores(dataset):
    # Fixing numerical columns that have _ in them
    numerical_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]

    for col in numerical_columns:
        if dataset[col].dtype == 'object':  # Check if the column contains string values
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')


def define_values(dataset):
    # We are predicting the credit score
    y = dataset['Credit_Score'].values
    # Remove features that don't help us predict credit score
    X = dataset.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Credit_Score', 'Month', 'Type_of_Loan', 'Credit_History_Age']).values

    # X_train, y_train = dataset['x'], dataset['y']
    # X_test, y_test = test_df['x'], test_df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

def scale(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def random_forest(X_train, X_test, y_train, y_test):
    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    # Train the model on the training data
    rf_classifier.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, y_pred)
    return report

train_df = pd.read_csv('data/train.csv', dtype = 'object')

train_df = show_stats(train_df)
train_df = formatting(train_df)
train_df = interpolate_missing_values(train_df)
train_df = convert_objects_to_labels(train_df)
train_df = remove_underscores(train_df)
X_train, X_test, y_train, y_test = define_values(train_df)
X_train, X_test = scale(X_train, X_test)
print(random_forest(X_train, X_test, y_train, y_test))