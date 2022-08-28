from ai_utec import gradient_descent
from ai_utec import utility
import pandas as pd
import numpy as np

x, y = utility.read_csv(path = 'data/iris.csv',
                        x_cols = ['sepal_length','sepal_width'],
                        y_cols = ['species'])

indices = list(map(lambda e : e in ['setosa', 'versicolor'], y))

x = x[indices, :]
y = y[indices, :]

print(x)
print(y)