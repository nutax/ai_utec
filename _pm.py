from ai_utec import gradient_descent
from ai_utec import utility
import numpy as np
import pandas as pd


x = np.arange(0,2*np.pi,0.2).reshape((-1, 1))
y =  np.array([ np.sin(e + np.random.normal(0,0.1) ) for e  in x]).reshape((-1, 1))

print(x)


x_train, y_train, x_val, y_val, x_test, y_test = utility.classic_split(x,y)

w, predict, loss_train, loss_val = gradient_descent.train(x_train = x_train,
                                                          y_train = y_train,
                                                          x_val = x_val,
                                                          y_val = y_val,
                                                          epochs = 100000,
                                                          alpha = 0.001,
                                                          norm_f = utility.norm,
                                                          denorm_f = utility.denorm,
                                                          extra_f = utility.poly_matrix_gen(16),
                                                          winit_f = utility.random_weights,
                                                          hypo_f = utility.weighted_sum,
                                                          diff_f = utility.difference,
                                                          loss_f = utility.lm_loss,
                                                          delta_f = utility.lm_delta,
                                                          update_f = utility.substract_delta,
                                                          batch_f = utility.random_batch_gen(4))

print(utility.r_squared(y, predict(x)))