import numpy as np
import pandas as pd

def read_csv(path, x_cols, y_cols):
        dataset = pd.read_csv(path)
        x = (dataset[x_cols]).to_numpy()
        y = (dataset[y_cols]).to_numpy()
        return x, y

def split(x, y, p):
        m = x.shape[0]
        train_p = int(m*p)
        indices = np.random.permutation(m)
        test_i, train_i = indices[:train_p], indices[train_p:]
        x1, x2 = x[train_i,:], x[test_i,:]
        y1, y2 = y[train_i,:], y[test_i,:]
        return x1, y1, x2, y2

def norm(x, xmin, xmax):
        return (x-xmin)/(xmax-xmin)


def denorm(x, xmin, xmax):
        return x*(xmax-xmin) + xmin