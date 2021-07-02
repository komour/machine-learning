import time
from random import randrange


def main():
    start_time = time.time()
    # f = open('resources/SVM/input.txt')
    n = int(input())
    # n = int(f.readline())
    threshold = 5
    b = 0
    eps = 1e-9
    exec_time = 2.87
    lambdas = [0 for _ in range(n)]
    kernel_values = [[] for _ in range(n)]
    classes = []
    for i in range(n):
        line = list(map(int, input().split()))
        # line = list(map(int, f.readline().split()))
        classes.append(line[-1])
        line.pop()
        for k in line:
            kernel_values[i].append(k)
    regularization_parameter = int(input())
    # regularization_parameter = int(f.readline())

    passed_amount = 0
    while passed_amount < threshold and time.time() - start_time < exec_time:
        changed_lambdas_amount = 0
        for i in range(n):

            #  Calculate E_i = f(x_i) - y_i
            E_i = -classes[i] + b
            for k in range(n):
                E_i += lambdas[k] * classes[k] * kernel_values[k][i]

            if not (classes[i] * E_i < -eps) and lambdas[i] < regularization_parameter \
                    or (classes[i] * E_i > eps) and lambdas[i] > 0:
                continue

            j = randrange(n)

            #  Calculate E_j = f(x_j) - y_j
            E_j = -classes[j] + b
            for k in range(len(classes)):
                E_j += lambdas[k] * classes[k] * kernel_values[k][j]

            lmbd_i_old = lambdas[i]
            lmbd_j_old = lambdas[j]

            if classes[i] != classes[j]:
                L = max(0.0, lambdas[j] - lambdas[i])
                H = min(regularization_parameter, regularization_parameter + lambdas[j] - lambdas[i])
            else:
                L = max(0, lambdas[i] + lambdas[j] - regularization_parameter)
                H = min(regularization_parameter, lambdas[i] + lambdas[j])

            if L == H:
                continue

            #  η = 2 * <x_i, x_j> - <x_i, x_i> - <x_j, x_j>
            eta = 2 * kernel_values[i][j] - kernel_values[i][i] - kernel_values[j][j]

            if eta >= 0:
                continue

            lambdas[j] -= classes[j] * (E_i - E_j) / eta
            lambdas[j] = H if lambdas[j] > H else L if lambdas[j] < L else lambdas[j]

            if abs(lambdas[j] - lmbd_j_old) < eps:
                continue

            lambdas[i] += classes[i] * classes[j] * (lmbd_j_old - lambdas[j])

            b1 = b - E_i - classes[i] * (lambdas[i] - lmbd_i_old) * kernel_values[i][i] - \
                 classes[j] * (lambdas[j] - lmbd_j_old) * kernel_values[i][j]
            b2 = b - E_j - classes[i] * (lambdas[i] - lmbd_i_old) * kernel_values[i][j] - \
                 classes[j] * (lambdas[j] - lmbd_j_old) * kernel_values[j][j]

            b = b1 if 0 < lambdas[i] < regularization_parameter \
                else b2 if 0 < lambdas[j] < regularization_parameter \
                else (b1 + b2) / 2.0

            changed_lambdas_amount += 1
        if changed_lambdas_amount == 0:
            passed_amount += 1

    # print(time.time() - start_time)
    for lmbd in lambdas:
        print(lmbd if lmbd > 0 else 0)
    print(b)


if __name__ == '__main__':
    main()
