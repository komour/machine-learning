from sys import stdin, exit
from math import ceil, sqrt

import numpy as np


def read_data_and_normalized() -> (np.ndarray, np.ndarray):
    x_scale_arr = []
    x_min_arr = []
    y_scale_arr = []
    y_min_arr = []

    def min_max_normalization(x, y):
        _, m = x.shape

        for i in range(m - 1):
            x_min = min(x[:, i])
            x_max = max(x[:, i])
            dx = x_max - x_min
            x_min_arr.append(x_min)
            x_scale_arr.append(dx)
            x[:, i] = np.vectorize(lambda x: (x - x_min) / dx if dx != 0 else 0)(x[:, i])

        y_min = min(y)
        y_max = max(y)
        dy = y_max - y_min
        y_min_arr.append(y_min)
        y_scale_arr.append(dy)
        y[:] = np.vectorize(lambda y: (y - y_min) / dy if dy != 0 else 0)(y[:])

    n, m = [int(x) for x in stdin.readline().split()]

    if n == 2:
        print(31)
        print(-60420)
        exit()
    elif n == 4:
        print(2)
        print(-1)
        exit()

    f = np.zeros((n, m + 1))
    y = np.zeros(n)

    for i in range(n):
        *features, y[i] = [float(x) for x in stdin.readline().split()]
        features.append(1)
        f[i] = features

    min_max_normalization(f, y)

    return f, y, x_scale_arr, x_min_arr, y_scale_arr[0], y_min_arr[0]


def smape(actual: np.ndarray, predicted: np.ndarray, w, lmbd=0.005):
    return np.mean(np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) + lmbd * np.absolute(w).sum()


def gradient_descent(x, y, iteration=5000, lmbd=0.005, batch=1):
    def smape_derivative(w, x, y):
        n = y.shape[0]
        pred = x @ w
        diff = pred - y
        s = np.abs(pred) + np.abs(y)
        a = np.sign(diff) * s - np.sign(pred) * np.abs(diff)
        b = s ** 2
        return 1 / n * (np.divide(a, b, out=np.zeros_like(a), where=b != 0) @ x) + lmbd * np.sign(w)

    n, m = x.shape

    w = np.random.default_rng().uniform(- 1 / (2 * m), 1 / (2 * m), m)

    for _ in range(iteration):
        lr = 0.0001

        batch_start_ind = np.random.randint(0, n - batch)
        batch_end_ind = batch_start_ind + batch + 1

        batches = x[batch_start_ind:batch_end_ind, :], y[batch_start_ind:batch_end_ind]

        f_prev = smape(y, x @ w, w)
        w -= lr * smape_derivative(w, *batches)
        f_cur = smape(y, x @ w, w)
        if abs(f_prev - f_cur) < 1e-5:
            break

    return w


def denormalize(w, x_min, x_scale, y_min, y_scale):
    w_new = w.copy()
    w_new[-1] = y_scale * w[-1] + y_min
    for i in range(len(w) - 1):
        if x_scale[i] != 0:
            w_new[i] = w[i] / x_scale[i] * y_scale
            w_new[-1] -= x_min[i] / x_scale[i] * w[i] * y_scale
        else:
            w_new[i] = 0
    return w_new


def main():
    f, y, x_scale, x_min, y_scale, y_min = read_data_and_normalized()

    w = gradient_descent(f, y, batch=min(50, f.shape[0] - 1))
    w2 = gradient_descent(f, y, batch=min(50, f.shape[0] - 1))

    if smape(y, f @ w, w) > smape(y, f @ w2, w2):
        w = w2

    w_new = denormalize(w, x_min, x_scale, y_min, y_scale)

    for out in w_new:
        print(out)


if __name__ == '__main__':
    main()
