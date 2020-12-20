import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import random

prefix = 'resources/DT/DT_txt/'


def read_txt(filename):
    print(filename)
    cur_file = Path(prefix + filename + '.txt')
    lines = cur_file.read_text().split('\n')
    lines.pop(0)
    lines.pop(0)
    ys = []
    xs = []
    for i in range(0, len(lines) - 1):
        xs.append(list(lines[i].split()))
        ys.append(xs[i].pop())
    return xs, ys


datasets = []
types = ["train", "test"]
for data_number in range(1, 22):
    cur_data = []
    for t in types:
        file_name = f"{data_number:02d}_" + t
        cur_data.append(read_txt(file_name))
    datasets.append(cur_data)
    # print(datasets[0][0][0][0], datasets[0][0][1][0])
    # idx1 - dataset number | idx2 - 0=train, 1=test | idx3 - 0=xs, 1=ys | idx4 - number of x or y

criteria = ["gini", "entropy"]
splitters = ["best", "random"]
heights = [h for h in range(1, 21)]


def get_accuracy(data_number, criterion, splitter, height):
    train = datasets[data_number][0]
    test = datasets[data_number][1]
    decision_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=height)
    decision_tree.fit(train[0], train[1])
    return accuracy_score(test[1], decision_tree.predict(test[0]))


best_params = [{} for _ in range(len(datasets))]


def calculate_best_params():
    for data_number in range(len(datasets)):
        cur_best_accuracy = 0
        for criterion in criteria:
            for splitter in splitters:
                for max_depth in heights:
                    cur_accuracy = get_accuracy(data_number, criterion, splitter, max_depth)
                    if cur_accuracy > cur_best_accuracy:
                        cur_best_accuracy = cur_accuracy
                        best_params[data_number] = {'dataset_number': data_number + 1,
                                                    'accuracy': cur_accuracy,
                                                    'criterion': criterion,
                                                    'splitter': splitter,
                                                    'max_depth': max_depth}


for d in best_params:
    print(d)

min_max_depth = 1
max_max_depth = 28

min_depth_data_number = 2
min_depth_best_criterion = 'gini'
min_depth_best_splitter = 'best'

max_depth_data_number = 20
max_depth_best_criterion = 'entropy'
max_depth_best_splitter = 'best'


def get_accuracy_train(data_number, criterion, splitter, height):
    train = datasets[data_number][0]
    decision_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=height)
    decision_tree.fit(train[0], train[1])
    return accuracy_score(train[1], decision_tree.predict(train[0]))


xs = [h for h in range(min_max_depth, max_max_depth)]
ys_max = []
ys_max_train = []
ys_min = []
ys_min_train = []
for max_depth in xs:
    ys_min.append(get_accuracy(min_depth_data_number, min_depth_best_criterion, min_depth_best_splitter, max_depth))
    ys_min_train.append(
        get_accuracy_train(min_depth_data_number, min_depth_best_criterion, min_depth_best_splitter, max_depth))
    ys_max.append(get_accuracy(max_depth_data_number, max_depth_best_criterion, max_depth_best_splitter, max_depth))
    ys_max_train.append(
        get_accuracy_train(max_depth_data_number, max_depth_best_criterion, max_depth_best_splitter, max_depth))


def draw_plot(xs, ys, title):
    plt.plot(xs, ys)
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.show()


draw_plot(xs, ys_min, 'dataset #3 test')
draw_plot(xs, ys_min_train, 'dataset #3 train')
draw_plot(xs, ys_max, 'dataset #21 test')
draw_plot(xs, ys_max_train, 'dataset #21 train')


# data - dataset[i][j] (train / test lvl)
def get_random_subset(data, size):
    xs = []
    ys = []
    for _ in range(size):
        idx = random.randint(0, len(data[0]) - 1)
        indices.append(ids)
        xs.append(data[0][idx])
        ys.append(data[1][idx])
    return xs, ys


forest = []


def predict_one_object(xs):
    predictions = {}
    for tree in forest:
        prediction = tree.predict_with_forest(xs)
        if prediction not in predictions:
            predictions[prediction] = 1
        else:
            predictions[prediction] += 1
    res_prediction = max(predictions, key=predictions.get)
    return res_prediction


def predict(xs):
    ys = []
    for x in xs:
        ys.append(predict_one_object(x))
    return ys


def create_forest(data_number, trees_count, size, criterion, splitter):
    forest.clear()
    train = datasets[data_number][0]
    for _ in range(trees_count):
        tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
        xs, ys = get_random_subset(train, size)
        tree.fit(xs, ys)
        forest.append(tree)


def get_forest_accuracy(data_number, criterion, splitter, is_train=False):
    train = datasets[data_number][0]
    test = datasets[data_number][1]
    trees_count = 10
    size = int(np.sqrt(len(train)))
    create_forest(data_number, trees_count, size, criterion, splitter)
    if is_train:
        return accuracy_score(predict(train[0]), train[1])
    else:
        return accuracy_score(predict(test[0]), test[1])


best_params_forest = [{} for _ in range(len(datasets))]
best_params_forest_train = [{} for _ in range(len(datasets))]


def calculate_best_params_forest():
    for data_number in range(len(datasets)):
        cur_best_accuracy = 0
        cur_best_accuracy_train = 0
        for criterion in criteria:
            for splitter in splitters:
                cur_accuracy = get_forest_accuracy(data_number, criterion, splitter)
                cur_accuracy_train = get_forest_accuracy(data_number, criterion, splitter, True)
                if cur_accuracy_train > cur_best_accuracy_train:
                    cur_best_accuracy_train = cur_accuracy_train
                    best_params_forest[data_number] = {'dataset_number': data_number + 1,
                                                       'accuracy': cur_accuracy,
                                                       'criterion': criterion,
                                                       'splitter': splitter}
                if cur_accuracy > cur_best_accuracy:
                    cur_best_accuracy = cur_accuracy
                    best_params_forest[data_number] = {'dataset_number': data_number + 1,
                                                       'accuracy': cur_accuracy,
                                                       'criterion': criterion,
                                                       'splitter': splitter}



for d in best_params_forest:
    print(d)

print('TRAIN')
for d in best_params_forest_train:
    print(d)


def main():
    calculate_best_params_forest()


if __name__ == '__main__':
    main()