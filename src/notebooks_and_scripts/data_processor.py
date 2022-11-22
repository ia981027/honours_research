from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split


def get_credit_ratings_train_test_split(test_size : float = 0.20):
    corporate_ratings = pd.read_csv('/home/vhutali/Desktop/projects/python/ML_finance_project_fast_api/resources'
                                    '/datasets/corporateCreditRatingWithFinancialRatios.csv')
    corporate_ratings = corporate_ratings.rename(
        columns={"Rating Agency": "RatingAgency", "Rating Date": "Date", "Ticker": "Symbol", "Corporation": "Name"})
    corporate_ratings["Date"] = pd.to_datetime(corporate_ratings["Date"])
    corporate_ratings["Date"] = corporate_ratings["Date"].dt.strftime('%Y-%m-%d')
    corporate_ratings = corporate_ratings[
        corporate_ratings["RatingAgency"] == "Standard & Poor's Ratings Services"].reset_index(drop=True)
    ratings = corporate_ratings["Rating"]
    ratings = [rating.replace("-", "").replace("+", "") for rating in ratings]
    ratings = LabelEncoder().fit_transform(ratings)
    print(f"Number of unique ratings: {len(np.unique(ratings))}")
    corporate_ratings["Sector"] = LabelEncoder().fit_transform(corporate_ratings["Sector"])
    corporate_ratings = corporate_ratings.drop(["Name", "RatingAgency", "Symbol", "SIC Code", "Date", "CIK", "Binary Rating", "Rating"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(np.array(corporate_ratings, dtype=np.float64), ratings,
                                                        test_size=0.15, random_state=0)
    x_train, x_validation, y_train, y_validation = train_test_split(np.array(x_train, dtype=np.float64), y_train,
                                                        test_size=0.17, random_state=0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test, y_train, y_test, x_validation, y_validation, corporate_ratings.columns


def get_pct_change_train_test_split(test_size : float = 0.20):
    pct_change = pd.read_csv('/home/vhutali/Desktop/projects/python/ML_finance_project_fast_api/resources/datasets/pct_change.csv', index_col="index")
    pct_change = pct_change.drop(columns=["Unnamed: 0"]).dropna()
    y = pct_change["change"]
    x = pct_change.drop(columns=["change"])
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64), test_size=0.15, random_state=0)
    x_train, x_validation, y_train, y_validation = train_test_split(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64), test_size=0.17, random_state=0)
    return x_train, x_test, y_train, y_test,  x_validation, y_validation, pct_change.columns
