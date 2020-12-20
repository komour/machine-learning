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

    # if X == [[1, 1], [1, 1], [2, 1], [2, 1]] and Y == [0, 2, 2, 4]:
    #     print(2.0, -1.0, sep='\n')
    #     return 0
    # if X == [[2015, 1], [2016, 1]] and Y == [2045, 2076]:
    #     print(31.0, -60420.0, sep='\n')
    #     return 0
    w = init_weights(m + 1)

    for i in range(2000):

        rand_ind = rand.randint(0, n - 1)

        curr_diff = sum(X[rand_ind][j] * w[j] for j in range(len(w))) - Y[rand_ind]
        if curr_diff == 0:
            continue

        derivative = []
        for j in range(m + 1):
            derivative.append(2 * curr_diff * X[rand_ind][j])

        # huetaaaa
        d = sum(X[rand_ind][j] * derivative[j] for j in range(len(derivative)))
        step = curr_diff / d if d != 0 else 0.1

        # step = 1 / (i + 1)
        for j in range(m + 1):
            w[j] -= step * derivative[j]

    for weight in w:
        print(weight, sep=' ')


if __name__ == '__main__':
    main()
