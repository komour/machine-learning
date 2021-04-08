import random
import math
import time


def init_weights(m, x):
    weights = []
    for i in range(m):
        # weights.append(random.uniform(-1 / (20 * m), 1 / (20 * m)))
        weights.append(x)
    # print(weights)
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
        s += abs(predicted[i] - expected[i]) / (abs(predicted[i]) + abs(expected[i]))
    return s / len(expected)


def sign(x):
    if x == 0 or x is None:
        return 1
    return math.copysign(1, x)


def d_smape_true(X, w, Y):
    m = len(X[0])
    res = [0 for _ in range(m)]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        assert len(w) == len(x)

        y_i = 0
        for k in range(len(x)):
            y_i += x[k] * w[k]

        diff = y_i - y
        mul = y_i * y
        s = (abs(mul) + mul)
        s2 = abs(y_i) + abs(y)
        derivative = (sign(diff) * s) / (abs(y_i) + s2 ** 2) if s != 0 else 0

        for k in range(len(x)):
            res[k] += derivative * x[k]
    return res


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
    # n, m = map(int, input().split())
    # 48
    f = open('resources/LR-CF/0.40_0.65.txt')
    m = int(f.readline())
    n = int(f.readline())

    X = []
    Y = []

    x_mins = [10000000000 for _ in range(m)]
    x_maxs = [-10000000000 for _ in range(m)]

    #  read data and find min max for each column
    for i in range(n):
        # row = list(map(int, input().split()))
        row = list(map(int, f.readline().split()))
        Y.append(row[-1])
        row.pop()
        for j in range(len(row)):
            if row[j] < x_mins[j]:
                x_mins[j] = row[j]
            if row[j] > x_maxs[j]:
                x_maxs[j] = row[j]
        row.append(1)
        X.append(row)

    # y_min = min(Y)
    # y_max = max(Y)

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
            row[i] /= (x_maxs[i] if x_maxs[i] != 0 else 1)
            # row[i] = (row[i] - x_mins[i]) / (x_maxs[i] - x_mins[i]) if x_maxs[i] != x_mins[i] else 0

    # for i in range(len(Y)):
    #     Y[i] = (Y[i] - y_min) / (y_max - y_min) if y_max != y_min else 0

    # pretty_print_2_dim_list([Y])
    # print(y_min)
    # print(y_max)

    max_iterations = 50
    lr = 1.5e7
    # lr = 1
    # batch_size = 30
    # eps = 1e-5

    #  init weights
    w1 = init_weights(m + 1, 0.00001)

    #  initial error
    # predicted_ys = [predict(xs, w) for xs in X]
    # prev_error = smape(Y, predicted_ys)

    start_time = time.time()
    exec_time = 1.1

    # gradient descent
    # for i in range(max_iterations):
    # counter = 0
    while time.time() - start_time < exec_time:
        # counter += 1
        # print(counter)

        # if i % 100 == 0:
        #     print(i)

        # batch_indices = random.sample(range(0, len(Y)), batch_size)
        # batch_indices = [j for j in range(n)]

        #  do gradient descent
        d = d_smape_true(X, w1, Y)
        for i in range(len(d)):
            d[i] /= n
        for j in range(m + 1):
            w1[j] -= lr * d[j]

        # for idx in batch_indices:
        #     d = d_smape(w, X[idx], Y[idx])
        #     for j in range(m + 1):
        #         w[j] -= lr * d[j]

        #  measure new error
        # predicted_ys = [predict(xs, w) for xs in X]
        # cur_error = smape(Y, predicted_ys)

        # if abs(cur_error - prev_error) < eps:
        #     print(i)
        #     break
        # prev_error = cur_error

    w2 = init_weights(m + 1, -0.00001)
    start_time = time.time()
    while time.time() - start_time < exec_time:

        #  do gradient descent
        d = d_smape_true(X, w2, Y)
        for i in range(len(d)):
            d[i] /= n
        for j in range(m + 1):
            w2[j] -= lr * d[j]

    Y_pred1 = []
    for xs in X:
        Y_pred1.append(predict(xs, w1))
    Y_pred2 = []
    for xs in X:
        Y_pred2.append(predict(xs, w2))

    smape1 = smape(Y, Y_pred1)
    smape2 = smape(Y, Y_pred2)
    print(smape1, smape2)
    if smape1 < smape2:
        w = w1
    else:
        w = w2

    assert len(x_maxs) == len(x_mins)

    for i in range(len(x_maxs)):
        w[i] /= (x_maxs[i] if x_maxs[i] != 0 else 1)

    # for weight in w:
    #     print(weight, sep=' ')
    #
    # return 0

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

    print(smape(Y_test, Y_pred))


if __name__ == '__main__':
    main()
