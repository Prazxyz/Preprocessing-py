import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datagolf.csv')
x = dataset.iloc[:,  :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[;, 1;3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder())])
