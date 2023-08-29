import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

train_df = pd.read_csv('data/train.csv', dtype = 'object')

for col in train_df.columns:
    print(f"Column '{col}' has data type: {train_df[col].dtype}")

numerical_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]

# Remove underscores from numerical columns
for col in numerical_columns:
    train_df[col] = train_df[col].str.replace('_', '')

# Specify desired data types for each column
dtypes = {'ID': object, 'Customer_ID': object, 'Month': object, 'Name': object, 'Age': int, 'SSN': object, 'Occupation': object, 'Annual_Income': float, 'Monthly_Inhand_Salary': float, 'Num_Bank_Accounts': int, 'Num_Credit_Card': int, 'Interest_Rate': int, 'Num_of_Loan': int, 'Type_of_Loan': object, 'Delay_from_due_date': int, 'Num_of_Delayed_Payment': int, 'Changed_Credit_Limit': float, 'Num_Credit_Inquiries': int, 'Credit_Mix': object, 'Outstanding_Debt': float, 'Credit_Utilization_Ratio': float, 'Credit_History_Age': int, 'Payment_of_Min_Amount': bool, 'Total_EMI_per_month': float, 'Amount_invested_monthly': float, 'Payment_Behaviour': object, 'Monthly_Balance': float, 'Credit_Score': object}


for col, dtype in dtypes.items():
    train_df[col] = train_df[col].astype(dtype)

# Remove negative values
for col in numerical_columns:
    train_df = train_df[train_df[col] >= 0]

train_df = train_df.drop_duplicates()
# train_df = train_df.dropna(inplace=True)
print(train_df.describe())

columns_with_nan = train_df.columns[train_df.isna().any()].tolist()
print(columns_with_nan)


print(train_df.head())


# Specify the names of the categorical columns and converting them to labels
categorical_columns = ["Month", "Occupation", "Type_of_Loan", "Credit_Mix", "Payment_Behaviour", "Payment_of_Min_Amount"]

le = preprocessing.LabelEncoder()

for col_name in categorical_columns:
    col_index = train_df.columns.get_loc(col_name)  # Find the index of the column by column name
    train_df[col_name] = le.fit_transform(train_df[col_name])


# Fixing numerical columns that have _ in them
numerical_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]


for col in numerical_columns:
    if train_df[col].dtype == 'object':  # Check if the column contains string values
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')




# Check how many entries are 0
print(train_df.isna().sum())

# We are predicting the credit score
y = train_df['Credit_Score'].values
# Remove features that don't help us predict credit score
X = train_df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Credit_Score', 'Month', 'Type_of_Loan', 'Credit_History_Age']).values

# X_train, y_train = train_df['x'], train_df['y']
# X_test, y_test = test_df['x'], test_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)