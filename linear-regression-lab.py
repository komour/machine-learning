import random as rand
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Simple error
def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted


# Mean Squared Error
def mse(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.square(_error(actual, predicted)))


# Root Mean Squared Error
def rmse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(mse(actual, predicted))


# Normalized Root Mean Squared Error
def nrmse(actual: np.ndarray, predicted: np.ndarray):
    return rmse(actual, predicted) / (actual.max() - actual.min())


# Symmetric Mean Absolute Percentage Error
def smape(actual: np.ndarray, predicted: np.ndarray):
    EPSILON = 1e-10
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def least_squares(X, Y):
    l = 1e6
    _, m = X.shape
    I = np.eye(m)
    return inv((X.T.dot(X)) + l * I).dot(X.T).dot(Y)


def mult(a, b):
    assert len(a) == len(b)
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def init_weights(n):
    weights = []
    for i in range(n):
        weights.append(rand.uniform(-1 / (2 * n), 1 / (2 * n)))
    return weights


def calc_ls(X, Y):
    w_ls = list(least_squares(np.array(X), np.array(Y)))
    Y_test_np = np.array(Y)
    Y_predicted_ls = []
    for i in range(len(X)):
        Y_predicted_ls.append(mult(X[i], w_ls))
    Y_predicted_ls_np = np.array(Y_predicted_ls)
    nrmse_ls = nrmse(Y_test_np, Y_predicted_ls_np)
    smape_ls = 100 * smape(Y_test_np, Y_predicted_ls_np)
    return nrmse_ls, smape_ls


def main():
    input_file = open("resources/LR/1.txt")

    #  read and normalise data
    m = int(input_file.readline())
    n = int(input_file.readline())
    X_tmp = []
    X = []
    Y = []
    X_test = []
    X_test_tmp = []
    Y_test = []

    scaler = MinMaxScaler()

    for i in range(n):
        row = list(map(int, input_file.readline().split()))
        Y.append(row.pop())
        X_tmp.append(row)
    X_tmp = scaler.fit_transform(X_tmp)
    for i in range(n):
        X.append(list(np.append(X_tmp[i], 1)))

    n_test = int(input_file.readline())

    for i in range(n_test):
        row = list(map(int, input_file.readline().split()))
        Y_test.append(row.pop())
        row.append(1)
        X_test_tmp.append(row)

    nrmse_ls, smape_ls = calc_ls(X_test_tmp, Y_test)  # calc ls
    # print("Least Squares NRMSE: ", nrmse_ls)
    # print("Least Squares SMAPE: ", smape_ls)

    for i in range(n_test):
        X_test_tmp[i].pop()
    X_test_tmp = scaler.fit_transform(X_test_tmp)
    for i in range(n_test):
        X_test.append(list(np.append(X_test_tmp[i], 1)))

    Y_test_np = np.array(Y_test)
    Y_np = np.array(Y)

    w = init_weights(m + 1)

    plot_data_nrmse = []
    plot_data_nrmse_train = []
    plot_data_smape = []
    plot_data_smape_train = []

    tau = 0.7
    lmbd = 0.9  # темп забывания предыстории ряда
    # iterations = [i for i in range(1, 200)] + [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    # iterations = [i for i in range(1, 2000)]
    iterations = [500]
    # taus = [0.7, 0.9]
    # taus_data = []
    for iters in iterations:
        # for tau in taus:
        # init
        derivative = []
        rand_ind = rand.randint(0, n - 1)
        curr_diff = sum(X[rand_ind][j] * w[j] for j in range(len(w))) - Y[rand_ind]
        for j in range(m + 1):
            derivative.append(2 * curr_diff * X[rand_ind][j] + tau * w[j])  # ridge regularisation
        for i in range(iters):
            rand_ind = rand.randint(0, n - 1)
            curr_diff = sum(X[rand_ind][j] * w[j] for j in range(len(w))) - Y[rand_ind]
            for j in range(m + 1):
                derivative[j] = derivative[j] * (1 - lmbd) + lmbd * 2 * curr_diff * X[rand_ind][j] + tau * w[j]
            # h = 1 / (i + 1)
            h = 0.01
            for j in range(m + 1):
                w[j] = w[j] * (1 - h * tau) - h * derivative[j]  # ridge regularisation
                # w[j] -= h * derivative[j]

        # for weight in w:
        #     print(weight, sep=' ')
        Y_predicted = []
        Y_predicted_train = []

        for i in range(len(X_test)):
            Y_predicted.append(mult(X_test[i], w))
        for i in range(len(X)):
            Y_predicted_train.append(mult(X[i], w))

        Y_predicted_np = np.array(Y_predicted)
        Y_predicted_train_np = np.array(Y_predicted_train)
        print(nrmse(Y_test_np, Y_predicted_np))
        print(100 * smape(Y_test_np, Y_predicted_np))

        plot_data_nrmse.append(nrmse(Y_test_np, Y_predicted_np))
        plot_data_smape.append(100 * smape(Y_test_np, Y_predicted_np))

        # taus_data.append(100 * smape(Y_test_np, Y_predicted_np))

        plot_data_nrmse_train.append(nrmse(Y_np, Y_predicted_train_np))
        plot_data_smape_train.append(100 * smape(Y_np, Y_predicted_train_np))

    # plt.plot(taus, taus_data)
    # plt.xlabel("tau")
    # plt.ylabel("smape")
    # plt.show()

    plt.plot(iterations, plot_data_smape)
    plt.xlabel("iterations")
    plt.ylabel("smape")
    plt.show()

    plt.plot(iterations, plot_data_nrmse)
    plt.xlabel("iterations")
    plt.ylabel("nrmse")
    plt.show()

    plt.plot(iterations, plot_data_smape_train)
    plt.xlabel("TRAIN-iterations")
    plt.ylabel("smape")
    plt.show()

    plt.plot(iterations, plot_data_nrmse_train)
    plt.xlabel("TRAIN-iterations")
    plt.ylabel("nrmse")
    plt.show()

    print("Least Squares NRMSE: ", nrmse_ls)
    print("Least Squares SMAPE: ", smape_ls)


if __name__ == '__main__':
    main()
