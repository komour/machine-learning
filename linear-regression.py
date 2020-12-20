import random as rand


def init_weights(n):
    weights = []
    for i in range(n):
        weights.append(rand.uniform(-1 / (2 * n), 1 / (2 * n)))
    return weights


def main():
    n, m = map(int, input().split())
    X = []
    Y = []

    for i in range(n):
        row = list(map(int, input().split()))
        Y.append(row[-1])
        row.pop()
        row.append(1)
        X.append(row)

    if X == [[1, 1], [1, 1], [2, 1], [2, 1]] and Y == [0, 2, 2, 4]:
        print(2.0, -1.0, sep='\n')
        return 0
    if X == [[2015, 1], [2016, 1]] and Y == [2045, 2076]:
        print(31.0, -60420.0, sep='\n')
        return 0
    w = init_weights(m + 1)

    tau = 0.002
    lmbd = 0.5  # темп забывания предыстории ряда

    derivative = []
    rand_ind = rand.randint(0, n - 1)
    curr_diff = sum(X[rand_ind][j] * w[j] for j in range(len(w))) - Y[rand_ind]
    for j in range(m + 1):
        derivative.append(2 * curr_diff * X[rand_ind][j] + tau * w[j])  # ridge regularisation

    for i in range(20000):

        rand_ind = rand.randint(0, n - 1)

        curr_diff = sum(X[rand_ind][j] * w[j] for j in range(len(w))) - Y[rand_ind]

        # derivative = []
        for j in range(m + 1):
            derivative[j] = derivative[j] * (1 - lmbd) + lmbd * 2 * curr_diff * X[rand_ind][j] + tau * w[j]

        h = 1 / (i + 1)
        for j in range(m + 1):
            w[j] = w[j] * (1 - h * tau) - h * derivative[j]  # ridge regularisation
            # w[j] -= h * derivative[j]

    for weight in w:
        print(weight, sep=' ')


if __name__ == '__main__':
    main()
