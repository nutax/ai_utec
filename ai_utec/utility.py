import numpy as np

def split_df(x, y, p):
        m = x.shape[0]
        train_p = int(m*p)
        indices = np.random.permutation(m)
        test_i, train_i = indices[:train_p], indices[train_p:]
        x1, x2 = x[train_i,:], x[test_i,:]
        y1, y2 = y[train_i,:], y[test_i,:]
        return x1, y1, x2, y2