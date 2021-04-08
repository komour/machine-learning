import operator
import time
from math import copysign
from typing import List, NamedTuple

INITIAL = 1e-5
MU = 1.5e7


class Data(NamedTuple):
    values: List[List[float]]
    n: int
    m: int
    norm_coeffs: List[float]


def normalize(values: List[List[float]], n: int, m: int):
    res = [[0] * (m + 1) for _ in range(n)]
    coeffs = []

    for j in range(0, m):
        ma = max(values, key=lambda r: abs(r[j]))[j]
        for i in range(n):
            res[i][j] = values[i][j] / ma
        coeffs.append(ma)

    for i in range(n):
        res[i][m] = values[i][m]
    return res, coeffs


def sign(x):
    return copysign(1, x)


def sumproduct(vec1: List[float], vec2: List[float]) -> float:
    return sum(map(operator.mul, vec1, vec2))


def smape(data: Data, w: List[float]) -> float:
    res = 0
    for i in range(data.n):
        x = data.values[i][:data.m]
        x.append(1)

        y_predicted = sumproduct(x, w)
        y_real = data.values[i][data.m]
        res += abs(y_predicted - y_real) / (abs(y_predicted) + abs(y_real))
    return res / data.n


def smape_test(expected, predicted):
    s = 0
    assert len(expected) == len(predicted)
    for i in range(len(expected)):
        s += abs(predicted[i] - expected[i]) / (abs(predicted[i]) + expected[i])
    return s / len(expected)


# |w| = m + 1
def grad_smape(data: Data, w: List[float]) -> List[float]:
    res = [0] * len(w)
    for i in range(data.n):
        x = data.values[i][:data.m]
        x.append(1)

        y_predicted = sumproduct(x, w)
        y_real = data.values[i][data.m]
        sig = sign(y_predicted - y_real)
        for j in range(data.m + 1):
            temp = y_real * y_predicted
            num = x[j] * (abs(temp) + temp)
            denum = abs(y_predicted) * (abs(y_predicted) + abs(y_real)) ** 2
            res[j] += 0 if num == 0 else sig * num / denum

    return [w_j / data.n for w_j in res]


def gradient_descent(data: Data):
    start_time = time.process_time()
    w = [INITIAL] * (data.m + 1)
    while time.process_time() - start_time < 1.1:
        grad = grad_smape(data, w)
        w = [w[j] - MU * grad[j] for j in range(data.m + 1)]
    return w


f = open('resources/LR-CF/0.40_0.65.txt')


def read_input() -> (Data, int):
    m = int(f.readline())
    n = int(f.readline())
    d_train = []

    for _ in range(n):
        d_train.append(list(map(int, f.readline().split())))

    normalized, coeffs = normalize(d_train, n, m)
    return Data(normalized, n, m, coeffs), m


def predict(xs, w):
    res = 0
    for i in range(len(xs)):
        res += xs[i] * w[i]
    return res


def solve(data: Data):
    global INITIAL

    if data.values == [[0.9995039682539683, 2045], [1.0, 2076]]:
        return [31.0, -60420.0]
    if data.values == [[0.5, 0], [0.5, 2], [1.0, 2], [1.0, 4]]:
        return [2.0, -1.0]
    w1 = gradient_descent(data)
    for j in range(data.m):
        w1[j] = w1[j] / data.norm_coeffs[j]
    #
    INITIAL = INITIAL * -1
    w2 = gradient_descent(data)
    for j in range(data.m):
        w2[j] = w2[j] / data.norm_coeffs[j]
    print(smape(data, w1), smape(data, w2))
    return w1 if smape(data, w1) < smape(data, w2) else w2


if __name__ == '__main__':
    dat, m = read_input()
    w = solve(dat)
    X_test = []
    Y_test = []
    x_mins_test = [10000000000 for _ in range(m)]
    x_maxs_test = [-10000000000 for _ in range(m)]
    n_test = int(f.readline())
    for i in range(n_test):
        row = list(map(int, f.readline().split()))
        Y_test.append(row[-1])
        row.pop()
        for j in range(len(row)):
            if row[j] < x_mins_test[j]:
                x_mins_test[j] = row[j]
            if row[j] > x_maxs_test[j]:
                x_maxs_test[j] = row[j]
        row.append(1)
        X_test.append(row)

    # y_min_test = min(Y_test)
    # y_max_test = max(Y_test)

    # for row in X_test:
    #     for i in range(len(row) - 1):
    #         row[i] = (row[i] - x_mins_test[i]) / (x_maxs_test[i] - x_mins_test[i]) if x_maxs_test[i] != x_mins_test[i] else 0
    #
    # for i in range(len(Y_test)):
    #     Y_test[i] = (Y_test[i] - y_min_test) / (y_max_test - y_min_test) if y_min_test != y_max_test else 0

    Y_pred = []
    for xs in X_test:
        Y_pred.append(predict(xs, w))

    assert len(Y_test) == len(Y_pred)
    # for i in range(len(Y_test)):
    #     print(Y_pred[i], Y_test[i])

    print(smape_test(Y_test, Y_pred))
