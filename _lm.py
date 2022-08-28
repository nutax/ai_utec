from ai_utec import gradient_descent
from ai_utec import utility

x, y = utility.read_csv(path = 'data/year_popu_empl.csv',
                        x_cols = ['Year', 'Population'],
                        y_cols = ['Employed'])


x_train, y_train, x_vt, y_vt = utility.random_split(x = x, y = y, p = 0.7)

x_val, y_val, x_test, y_test = utility.random_split(x = x_vt, y = y_vt, p = 0.66)

w, predict, loss_train, loss_val = gradient_descent.train(x_train = x_train,
                                                       y_train = y_train,
                                                       x_val = x_val,
                                                       y_val = y_val,
                                                       epochs = 1000,
                                                       alpha = 0.01,
                                                       norm_f = utility.norm,
                                                       denorm_f = utility.denorm,
                                                       extra_f = utility.add_bias,
                                                       winit_f = utility.random_weights,
                                                       hypo_f = utility.weighted_sum,
                                                       diff_f = utility.difference,
                                                       loss_f = utility.lm_loss,
                                                       delta_f = utility.lm_delta,
                                                       update_f = utility.substract_delta,
                                                       batch_f = utility.random_batch_gen(2))


print(w)