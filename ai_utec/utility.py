import numpy as np
import pandas as pd

def read_csv(path, x_cols, y_cols):
        dataset = pd.read_csv(path)
        x = (dataset[x_cols]).to_numpy()
        y = (dataset[y_cols]).to_numpy()
        return x, y

def random_split(x, y, p):
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

def add_bias(x):
        bias_col = np.ones((x.shape[0],1))
        return np.append(bias_col, x, 1)

def random_weights(n):
        return np.random.rand(n)

def weighted_sum(x, w):
        return np.matmul(x,w.T)


def difference(ans, prd):
        return ans - prd

def lm_loss(x, y, w, diff):
        return np.dot(diff, diff)/diff.shape[0]

def lm_delta(x, y, w, diff):
        return np.dot(-x, diff)/diff.shape[0]

def substract_delta(w, dw, alpha):
        return w - alpha*dw

def random_batch_gen(size):
        def random_batch(x,y):
                indices = np.random.permutation(x.shape[0])
                batch_i = indices[:size]
                return x[batch_i,:], y[batch_i,:]
        return random_batch