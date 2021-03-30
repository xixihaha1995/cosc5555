import pandas as pd
import numpy as np
import copy

bank = pd.read_csv("bank.csv",delimiter=';')

for column in bank:
    if bank[column].dtypes == object:
        pd.get_dummies(bank[column], prefix=column)
# print(bank.dtypes)
# print(bank.columns)
# print(bank.head())
# print(bank.shape)
