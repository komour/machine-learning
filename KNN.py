import math


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
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev
}


def uniform(u):
    return 1 / 2 if abs(u) < 1 else 0


def triangular(u):
    return 1 - u if abs(u) < 1 else 0


def epanechnikov(u):
    return (3 / 4) * (1 - (u ** 2)) if abs(u) < 1 else 0


def quartic(u):
    return (15 / 16) * ((1 - (u ** 2)) ** 2) if abs(u) < 1 else 0


def triweight(u):
    return (35 / 32) * ((1 - (u ** 2)) ** 3) if abs(u) < 1 else 0


def tricube(u):
    return (70 / 81) * ((1 - (u ** 3)) ** 3) if abs(u) < 1 else 0


def gaussian(u):
    return (1 / (math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * (u ** 2))


def cosine(u):
    return (math.pi / 4) * math.cos(math.pi * u / 2) if abs(u) < 1 else 0


def logistic(u):
    return 1 / (math.exp(u) + 2 + math.exp(-u))


def sigmoid(u):
    return (2 / math.pi) / (math.exp(u) + math.exp(-u))


kernel_dict = {
    "uniform": uniform,
    "triangular": triangular,
    "epanechnikov": epanechnikov,
    "quartic": quartic,
    "triweight": triweight,
    "tricube": tricube,
    "gaussian": gaussian,
    "cosine": cosine,
    "logistic": logistic,
    "sigmoid": sigmoid
}


def read_int_list():
    return list(map(int, input().split()))


def nadaraya_watson(values_dists, h):
    denominator = 0
    numerator = 0
    s = 0
    for v, d in values_dists:
        k = kernel(d / h if h != 0 else d)
        print(v, k)
        denominator += k
        numerator += v * k
        s += v
    return numerator / denominator if denominator != 0 else s / len(values_dists)


nm = read_int_list()
n = nm[0]
m = nm[1]

X = []
Y = []

for j in range(n):
    row = read_int_list()
    Y.append(row[-1])
    row.pop()
    X.append(row)

R = read_int_list()
distance = distance_dict.get(input())
kernel = kernel_dict.get(input())
window_type = input()
arg = int(input())

distances = []
for point in X:
    distances.append(distance(point, R))

values_with_distances = list(zip(Y, distances))
values_with_distances.sort(key=lambda pair: pair[1])

if window_type == "variable":
    window_width = values_with_distances[arg][1]
else:
    window_width = arg


print(nadaraya_watson(values_with_distances, window_width))

