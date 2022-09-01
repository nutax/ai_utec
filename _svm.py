from ai_utec import gradient_descent
from ai_utec import utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x, y = utility.read_csv(path = 'data/iris.csv',
                        x_cols = ['sepal_length','sepal_width'],
                        y_cols = ['species'])

indices = list(map(lambda e : e in ['setosa', 'versicolor'], y))

x = x[indices, :]
y = y[indices, :]
y = np.apply_along_axis(lambda e : np.array([['setosa', 'versicolor'].index(e[0])]), 1, y)



x_train, y_train, x_val, y_val, x_test, y_test = utility.classic_split(x,y)

w, predict, loss_train, loss_val = gradient_descent.train(x_train = x_train,
                                                          y_train = y_train,
                                                          x_val = x_val,
                                                          y_val = y_val,
                                                          epochs = 10000,
                                                          alpha = 0.01,
                                                          norm_f = utility.norm,
                                                          denorm_f = utility.denorm,
                                                          extra_f = utility.add_bias,
                                                          winit_f = utility.random_weights,
                                                          hypo_f = utility.sigmoid_weighted_sum,
                                                          diff_f = utility.difference,
                                                          loss_f = utility.lm_loss,
                                                          delta_f = utility.lm_delta,
                                                          update_f = utility.substract_delta,
                                                          batch_f = utility.random_batch_gen(8))

y_prd = predict(x)

print(y.reshape(-1))
print(np.array([int(e) for e in np.round(y_prd.reshape(-1))]))