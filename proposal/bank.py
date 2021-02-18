import pandas as pd
import numpy as np
import copy
bankTrain = pd.read_csv("bank.csv",delimiter=';')
print(bankTrain.columns)
print(bankTrain.head())
print(bankTrain.shape)
