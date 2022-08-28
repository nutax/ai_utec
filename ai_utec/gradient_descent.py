import numpy as np

def train(x_train, y_train, x_val, y_val, epochs, alpha, norm_f, denorm_f, extra_f, winit_f, hypo_f, diff_f, loss_f, delta_f, update_f, batch_f):

    loss_tb, loss_vb = np.zeros(epochs), np.zeros(epochs)

    x_min, x_max = x_train.min(axis=0), x_train.max(axis=0)
    y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)

    x_tn, y_tn = norm_f(x_train, x_min, x_max), norm_f(y_train, y_min, y_max)
    x_vn, y_vn = norm_f(x_val, x_min, x_max), norm_f(y_val, y_min, y_max)

    x_tn = extra_f(x_tn)
    x_vn = extra_f(x_vn)

    w = winit_f()

    for i in range(epochs):
        x_tb, y_tb = batch_f(x_tn, y_tn)
        x_vb, y_vb = batch_f(x_vn, y_vn)

        diff_tb = diff_f(y_tb, hypo_f(x_tb, w))
        diff_vb = diff_f(y_vb, hypo_f(x_vb, w))

        dw = delta_f(x_tb, y_tb, w, diff_tb)
        w = update_f(w, dw, alpha)
        
        loss_tb[i] = loss_f(x_tb, y_tb, w, diff_tb)
        loss_vb[i] = loss_f(x_vb, y_vb, w, diff_vb)

    def predict(x):
        x_n = norm_f(x)
        x_n = extra_f(x_n)
        y_n = hypo_f(x_n, w)
        y = denorm_f(y_n)
        return y

    return predict, loss_tb, loss_vb


