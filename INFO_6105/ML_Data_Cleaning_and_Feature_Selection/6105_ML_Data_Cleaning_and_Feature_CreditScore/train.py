import argparse

import warnings
import pandas as pd
from pandas.api.types import is_numeric_dtype

import numpy as np

from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler as rbScaler
from sklearn.linear_model import LogisticRegression as lgrClassifier

warnings.filterwarnings('ignore')


def get_num_cols():
    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                'Total_EMI_per_month', 'Amount_invested_monthly',
                'Monthly_Balance', 'Credit_History_Age']

    return num_cols


def get_cat_cols():
    cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount',
                'Payment_Behaviour', 'Credit_Score']

    return cat_cols


def read_data():
    df = pd.read_csv('train.csv')
    return df


def take_years(x):
    if x is not None:
        return str(x).strip()[0:2]


def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) &
                    (df[name] < quant_df.loc[high, name])]

    return df


def clean_data(df, remove_outliers=True):
    irrelavent_coulumns = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
    df.drop(columns=irrelavent_coulumns, inplace=True, axis=1)

    df = df.applymap(
        lambda x: x if x is np.NaN or not
        isinstance(x, str) else str(x).strip('_')).replace(
        ['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN
    )

    df.Age = df.Age.astype(int)
    df.Annual_Income = df.Annual_Income.astype(float)
    df.Num_of_Loan = df.Num_of_Loan.astype(int)
    df.Num_of_Delayed_Payment = df.Num_of_Delayed_Payment.astype(float)
    df.Changed_Credit_Limit = df.Changed_Credit_Limit.astype(float)
    df.Outstanding_Debt = df.Outstanding_Debt.astype(float)
    df.Amount_invested_monthly = df.Amount_invested_monthly.astype(float)
    df.Monthly_Balance = df.Monthly_Balance.astype(float)

    df.Credit_History_Age = df.Credit_History_Age.apply(take_years)
    df['Credit_History_Age'] = df['Credit_History_Age'].replace({'na': np.NaN})
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace({'NM': 'No'})

    if remove_outliers:
        df = remove_outlier(df)

    df.interpolate(method='linear', inplace=True)

    return df


def fit_tranform(df):
    Occupation_le = le()
    Type_of_Loan_le = le()
    Credit_Mix_le = le()
    Credit_History_Age_le = le()
    Payment_of_Min_Amount_le = le()
    Payment_Behaviour_le = le()
    Credit_Score_le = le()

    df['Occupation'] = Occupation_le.fit_transform(df['Occupation'])
    df['Type_of_Loan'] = Type_of_Loan_le.fit_transform(df['Type_of_Loan'])
    df['Credit_Mix'] = Credit_Mix_le.fit_transform(df['Credit_Mix'])
    df['Credit_History_Age'] = Credit_History_Age_le.fit_transform(
        df['Credit_History_Age'])
    df['Payment_of_Min_Amount'] = Payment_of_Min_Amount_le.fit_transform(
        df['Payment_of_Min_Amount'])
    df['Payment_Behaviour'] = Payment_Behaviour_le.fit_transform(
        df['Payment_Behaviour'])
    df['Credit_Score'] = Credit_Score_le.fit_transform(df['Credit_Score'])

    return df


def create_splits(df):
    mdf = df[
        ['Credit_Score', 'Changed_Credit_Limit',
         'Payment_of_Min_Amount', 'Credit_Mix',
         'Delay_from_due_date', 'Annual_Income',
         'Age', 'Monthly_Balance',
         'Num_of_Delayed_Payment', 'Outstanding_Debt',
         'Payment_Behaviour', 'Credit_History_Age',
         'Num_Bank_Accounts'
         ]
    ]

    x = mdf.drop(['Credit_Score'], axis=1).values
    y = mdf['Credit_Score'].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=30)

    return x_train, x_test, y_train, y_test


def train(x_train, x_test, y_train, y_test):
    ro_scaler = rbScaler()
    x_train = ro_scaler.fit_transform(x_train)
    x_test = ro_scaler.fit_transform(x_test)
    lgr = lgrClassifier(C=100)
    lgr.fit(x_train, y_train)

    return lgr, x_train, x_test, y_train, y_test


def remove_and_impute_back(df, percent=10, fill_method='ffill'):
    percent_frac = percent / 100

    def delete(col):
        col.loc[col.sample(frac=percent_frac).index] = np.nan
        return col

    df.apply(delete, axis=0)
    if fill_method == "mode":
        df = df.fillna(df.mode().iloc[0])
    else:
        df = df.fillna(method=fill_method, axis=0)

        # This is to make sure, to not miss out the NA values
        # at the first and last rows of the columns
        if fill_method == "ffill":
            df = df.fillna(method='bfill', axis=0)
        elif fill_method == "bfill":
            df = df.fillna(method='ffill', axis=0)

    return df


def main(exclude_outliers=True, remove_percent=None,
         fillna_method=None, normal_process=False):
    assert type(exclude_outliers) == bool
    assert remove_percent is not None
    assert fillna_method is not None

    df = read_data()
    df = clean_data(df, exclude_outliers)
    df = fit_tranform(df)

    if not normal_process:
        df = remove_and_impute_back(df, percent=remove_percent, fill_method=fillna_method)

    x_train, x_test, y_train, y_test = create_splits(df)

    lgr, x_train, x_test, y_train, y_test = train(x_train, x_test, y_train, y_test)
    lgr_score = lgr.score(x_train, y_train)
    lgr_score_t = lgr.score(x_test, y_test)

    print(f"Train Score: {lgr_score}")
    print(f"Test Score: {lgr_score_t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-outliers",
        help="Pass this variable to exclude outliers in the data",
        default=True, action='store_true')
    parser.add_argument(
        "--include-outliers",
        help="Pass this variable to include outliers in the data",
        dest='exclude_outliers', action='store_false')
    parser.add_argument(
        "-r",
        "--remove-percent",
        help="Pass the number of percentage that you want to remove from the data")
    parser.add_argument(
        "-n",
        "--fillna-method",
        help="Pass the method that you want to use to fill the missing values")
    parser.add_argument(
        "--normal-process",
        help="Pass this variable to check the scores in the usual process",
        default=False, action='store_true')

    args = parser.parse_args()
    remove_percent = int(args.remove_percent)
    main(args.exclude_outliers, remove_percent, args.fillna_method, args.normal_process)
