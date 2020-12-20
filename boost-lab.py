import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import math
from random import random
import matplotlib.pyplot as plt

input_data_chips = pd.read_csv("resources/Boost/chips.csv")
chips_xs = input_data_chips[input_data_chips.columns[:-1]].to_numpy()
chips_ys = input_data_chips[input_data_chips.columns[-1]].to_numpy()
chips_ys = list(map(lambda c: c == 'P', chips_ys))

input_data_geyser = pd.read_csv("resources/Boost/geyser.csv")
geyser_xs = input_data_geyser[input_data_geyser.columns[:-1]].to_numpy()
geyser_ys = input_data_geyser[input_data_geyser.columns[-1]].to_numpy()
geyser_ys = list(map(lambda c: c == 'P', geyser_ys))


class Stump:
    def __init__(self, tree, say):
        self.decision_tree = tree
        self.say = say

    def predict(self, xs):
        return self.decision_tree.predict(xs)


class AdaForest:
    def __create_ada_stump__(self, sample_weight, max_depth=2):
        tree = DecisionTreeClassifier(max_depth=max_depth)
        sample_weight = sample_weight if sample_weight is not None else [1 / len(self.xs) for _ in range(len(self.xs))]
        tree.fit(self.xs, self.ys, sample_weight)

        # count total error to compute amount of say
        incorrect = []
        total_error = 0  # 1e-10
        for i in range(len(self.xs)):
            if tree.predict([self.xs[i]]) != self.ys[i]:
                total_error += sample_weight[i]
                incorrect.append(i)
        say = math.log((1 - total_error) / total_error) / 2

        # compute and normalize weights for the next stump
        new_weight = []
        for i in range(len(sample_weight)):
            if i in incorrect:  # increase weight of this sample if it's incorrectly predicted
                new_weight.append(sample_weight[i] * math.exp(say))
            else:
                new_weight.append(sample_weight[i] * math.exp(-say))  # vice versa
        return Stump(tree, say), list(map(lambda e: e / sum(new_weight), new_weight))

    def __init__(self, size, xs, ys, stumps_depth):
        self.forest = []
        self.xs = xs
        self.ys = ys
        cur_weights = None
        for _ in range(size):
            stump, new_weights = self.__create_ada_stump__(cur_weights, stumps_depth)
            cur_weights = new_weights
            self.forest.append(stump)

    def predict(self, xs):
        res = 0
        for stump in self.forest:
            if stump.predict([xs]):
                res += stump.say
            else:
                res -= stump.say
        return res >= 0


def draw_plot(xs, ys, title, xlabel, ylabel):
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# dataset = [(chips_xs, chips_ys, 'chips'), (geyser_xs, geyser_ys, 'geyser')]
# max_depths = [1, 2, 3, 4, 5]
# for data in dataset:
#     for max_depth in max_depths:
#         accuracy = []
#         for steps in range(1, 70):
#             forest = AdaForest(steps, data[0], data[1], max_depth)
#             predicted_ys = []
#             for xs in data[0]:
#                 cur_y = forest.predict(xs)
#                 predicted_ys.append(cur_y)
#             accuracy.append(accuracy_score(data[1], predicted_ys))
#         draw_plot([i for i in range(1, 70)], accuracy, f'{data[2]}, max_depth={max_depth}', 'amount of steps', 'accuracy')


def draw_classification_plot(x, y, title, stumps: list, launches=2, ncols=graph_width, h=.05):
    X0, X1 = x[:, 0], x[:, 1]
    x_min, x_max = X0.min() - 0.5, X0.max() + 0.5
    y_min, y_max = X1.min() - 0.5, X1.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = plt.plot(launches, ncols)
    ax.title.set_text(title)

    z = [[predict(stumps, np.array([i, j])) for i, j in zip(ii, jj)] for ii, jj in zip(xx, yy)]
    out = ax.contourf(xx, yy, np.array(z), cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.grid(True)


graph_index = 0

amount_of_steps = [1, 2, 3, 5, 8, 13, 21, 34, 55]

for i in range(len(datasets)):
    for d in range(len(depths)):
        for j in amount_of_steps:
            name, (xs, ys) = datasets[i]
            draw_classification_plot(xs,
                                     ys,
                                     f"{name}: {depths[d]} depth, {j} stumps",
                                     stumpList[d * len(datasets) + i][:j],
                                     len(datasets) * len(depths),
                                     ncols=len(amount_of_steps))