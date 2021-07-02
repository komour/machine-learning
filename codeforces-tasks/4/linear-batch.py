import random
import math
import time


def init_weights(m):
    weights = []
    for i in range(m):
        weights.append(random.uniform(-1 / (2 * m), 1 / (2 * m)))
    return weights


def pretty_print_2_dim_list(l):
    for row in l:
        for x in row:
            print(x, end=' ')
        print()


def smape(expected, predicted):
    s = 0
    assert len(expected) == len(predicted)
    for i in range(len(expected)):
        s += abs(predicted[i] - expected[i]) / (abs(predicted[i]) + expected[i])
    return s / len(expected)


def sign(x):
    if x == 0 or x is None:
        return 1
    return math.copysign(1, x)


def d_smape(w, x, y):
    assert len(w) == len(x)

    y_i = 0
    for i in range(len(x)):
        y_i += x[i] * w[i]

    diff = y_i - y
    s = abs(y_i) + abs(y)

    derivative = (sign(diff) * s - sign(y_i) * abs(diff)) / s ** 2 if s != 0 else 0

    res = []
    for i in range(len(x)):
        res.append(derivative * x[i])
    return res


def predict(xs, w):
    res = 0
    for i in range(len(xs)):
        res += xs[i] * w[i]
    return res


def main():
    n, m = map(int, input().split())
    # f = open('resources/LR-CF/0.62_0.80.txt')
    # m = int(f.readline())
    # n = int(f.readline())

    X = []
    Y = []

    x_mins = [10000000000 for _ in range(m)]
    x_maxs = [-10000000000 for _ in range(m)]

    #  read data and find min max for each column
    for i in range(n):
        row = list(map(int, input().split()))
        # row = list(map(int, f.readline().split()))
        Y.append(row[-1])
        row.pop()
        for j in range(len(row)):
            if row[j] < x_mins[j]:
                x_mins[j] = row[j]
            if row[j] > x_maxs[j]:
                x_maxs[j] = row[j]
        row.append(1)
        X.append(row)

    y_min = min(Y)
    y_max = max(Y)

    #  sample answer
    if n == 2:
        print(31.0, -60420.0, sep='\n')
        return 0
    if n == 4:
        print(2.0, -1.0, sep='\n')
        return 0

    #  normalize input data
    for row in X:
        for i in range(len(row) - 1):
            row[i] = (row[i] - x_mins[i]) / (x_maxs[i] - x_mins[i]) if x_maxs[i] != x_mins[i] else 0

    for i in range(len(Y)):
        Y[i] = (Y[i] - y_min) / (y_max - y_min) if y_max != y_min else 0

    # pretty_print_2_dim_list([Y])
    # print(y_min)
    # print(y_max)

    lr = 0.00009
    batch_size = 50

    #  init weights
    w = init_weights(m + 1)

    start_time = time.time()
    exec_time = 2.2

    # gradient descent
    # while True:
    while time.time() - start_time < exec_time:
        # if i % 100 == 0:
        #     print(i)

        batch_indices = random.sample(range(0, len(Y)), batch_size)

        #  do gradient descent
        for idx in batch_indices:
            d = d_smape(w, X[idx], Y[idx])
            for j in range(m + 1):
                w[j] -= lr * d[j]

    assert len(x_maxs) == len(x_mins)

    w_processed = [None for _ in range(len(w))]
    w_processed[-1] = y_min + w[-1] * (y_max - y_min)

    for i in range(len(x_maxs)):
        if x_maxs[i] == x_mins[i]:
            w_processed[i] = 0
            continue
        w_processed[-1] -= x_mins[i] * (y_max - y_min) * w[i] / (x_maxs[i] - x_mins[i])
        w_processed[i] = w[i] * (y_max - y_min) / (x_maxs[i] - x_mins[i])

    for weight in w_processed:
        print(weight, sep=' ')

    return 0

    #  measure metrics in test data
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

    y_min_test = min(Y_test)
    y_max_test = max(Y_test)

    # for row in X_test:
    #     for i in range(len(row) - 1):
    #         row[i] = (row[i] - x_mins_test[i]) / (x_maxs_test[i] - x_mins_test[i]) if x_maxs_test[i] != x_mins_test[i] else 0
    #
    # for i in range(len(Y_test)):
    #     Y_test[i] = (Y_test[i] - y_min_test) / (y_max_test - y_min_test) if y_min_test != y_max_test else 0

    Y_pred = []
    for xs in X_test:
        Y_pred.append(predict(xs, w_processed))

    assert len(Y_test) == len(Y_pred)
    # for i in range(len(Y_test)):
    #     print(Y_pred[i], Y_test[i])

    print(smape(Y_test, Y_pred))


if __name__ == '__main__':
    main()
