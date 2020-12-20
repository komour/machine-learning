import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f1(p, r):
    return 2 * p * r / (p + r)


def f_measure(k, matrix):
    row_sum = [0] * k
    column_sum = [0] * k
    true_positive = [0.0] * k
    amount = 0

    for i in range(k):
        row = matrix[i]
        for j in range(k):
            x = row[j]
            amount += x
            row_sum[i] += x
            column_sum[j] += x
            if i == j:
                true_positive[i] = x

    recall = [0.0] * k
    prec = [0.0] * k
    recall_sum = 0
    prec_sum = 0
    micro_sum = 0

    for i in range(k):
        if true_positive[i] == 0:
            continue
        prec[i] = float(true_positive[i]) / row_sum[i]
        prec_sum += prec[i] * row_sum[i]

        recall[i] = float(true_positive[i]) / column_sum[i]
        recall_sum += recall[i] * row_sum[i]

        micro_sum += f1(recall[i], prec[i]) * row_sum[i]

    prec_average = prec_sum / amount
    recall_average = recall_sum / amount

    macro = f1(prec_average, recall_average)
    micro = micro_sum / amount

    return macro


def manhattan(x, y):
    result = 0
    for i in range(len(x)):
        result += abs(x[i] - y[i])
    return result


def euclidean(x, y):
    result = 0
    for i in range(len(x)):
        result += (x[i] - y[i]) ** 2
    return math.sqrt(result)


def chebyshev(x, y):
    result = 0
    for i in range(len(x)):
        result = max(result, abs(x[i] - y[i]))
    return result


distance_dict = {
    manhattan: "manhattan",
    euclidean: "euclidean",
    chebyshev: "chebyshev"
}

distance_set = {manhattan, euclidean, chebyshev}


def uniform(u):
    return 1 / 2 if abs(u) < 1 else 0


def triangular(u):
    return 1 - u if abs(u) < 1 else 0


def epanechnikov(u):
    return (3 / 4) * (1 - (u ** 2)) if abs(u) < 1 else 0


def quartic(u):
    return (15 / 16) * ((1 - (u ** 2)) ** 2) if abs(u) < 1 else 0


kernel_dict = {
    uniform: "uniform",
    triangular: "triangular",
    epanechnikov: "epanechnikov",
    quartic: "quartic"
}

kernels = {uniform, triangular, epanechnikov, quartic}


def nadaraya_watson(values_dists, h):
    denominator = 0
    numerator = 0
    s = 0
    for val, dist in values_dists:
        k = kernel(dist / h if h != 0 else dist)
        denominator += k
        numerator += val * k
        s += val
    return numerator / denominator if denominator != 0 else s / len(values_dists)


def KNN_leave_one_out():
    results = []
    for index in range(len(X_init)):
        R = np.copy(X_init[index])
        X = np.copy(X_init[np.arange(len(X_init)) != index])
        Y = np.copy(Y_one_hot[np.arange(len(Y_one_hot)) != index])

        distances = []
        for point in X:
            distances.append(distance(point, R))

        values_with_distances = list(zip(Y, distances))
        values_with_distances.sort(key=lambda pair: pair[1])

        window_width = values_with_distances[neighbors_amount][1]

        res = nadaraya_watson(values_with_distances, window_width)
        current_max = max(res[np.arange(3)])
        maxes = []
        for i in range(3):
            if res[i] == current_max:
                maxes.append(i)
        results.append(random.choice(maxes) + 1)

    confusion_matrix = [[0] * 3, [0] * 3, [0] * 3]
    for i in range(len(results)):
        predicted_class = results[i] - 1
        actual_class = Y_init[i][0] - 1
        if results[i] == Y_init[i]:
            confusion_matrix[predicted_class][predicted_class] += 1
        else:
            confusion_matrix[predicted_class][actual_class] += 1
    return f_measure(3, confusion_matrix)


input_data = pd.read_csv("resources/dataset_wine.csv")
X_init = input_data[input_data.columns[1:]].to_numpy()
Y_init = input_data[input_data.columns[:1]].to_numpy()
D = len(X_init)

Y_one_hot = np.zeros(shape=(len(Y_init), 3))

i = 0
for v in Y_init:
    if v[0] == 1:
        Y_one_hot[i] = np.array([1, 0, 0])
    elif v[0] == 2:
        Y_one_hot[i] = np.array([0, 1, 0])
    elif v[0] == 3:
        Y_one_hot[i] = np.array([0, 0, 1])
    i += 1

max_f_measure = 0
best_distance = None
best_kernel = None
best_neighbors_amount = None
knn = 0
for distance in distance_set:
    for kernel in kernels:
        print()
        for neighbors_amount in range(1, int(math.sqrt(D))):
            cur_result = KNN_leave_one_out()
            print(cur_result, neighbors_amount, kernel_dict.get(kernel), distance_dict.get(distance), sep=' ', end='\n')
            if cur_result > max_f_measure:
                max_f_measure = cur_result
                best_distance = distance
                best_kernel = kernel
                best_neighbors_amount = neighbors_amount

print(distance_dict.get(best_distance), best_neighbors_amount, kernel_dict.get(best_kernel))
distance = best_distance
kernel = best_kernel
f_measures = []
for neighbors_amount in range(1, int(math.sqrt(D))):
    f_measures.append(KNN_leave_one_out())

plt.plot(range(1, int(math.sqrt(D))), f_measures)
plt.xlabel("Neighbors Amount")
plt.ylabel("F-measure")
plt.show()
