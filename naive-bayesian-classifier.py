import math

Q_dim = 2
eps = 1e-5


def _log(x):
    # if x == 0:
    #     return 0
    # else:
    #     return math.log(x)
    return math.log(x)


def _exp(x):
    # try:
    #     y = m.exp(x)
    # except OverflowError:
    #     y = float('inf')
    # return y
    return math.exp(x)


def main():
    # input_file = open("resources/cf-bayes/input.in")
    all_words = set()
    # class_amount = int(input_file.readline())
    class_amount = int(input())
    lmbd = list(map(float, input().split()))
    # lmbd = list(map(float, input_file.readline().split()))
    # alpha = int(input_file.readline())
    alpha = int(input())
    # n_train = int(input_file.readline())
    n_train = int(input())
    train_classes = []
    classes = [i for i in range(class_amount)]
    train_data = [set() for _ in range(n_train)]
    class_amount_with_word = {}
    amount_of_class = [0 for _ in classes]

    # read train data
    for i in range(n_train):
        # row = list(input_file.readline().split())
        row = list(input().split())
        cur_class = int(row.pop(0)) - 1
        train_classes.append(cur_class)
        amount_of_class[cur_class] += 1
        row.pop(0)
        for word in row:
            train_data[i].add(word)
            all_words.add(word)

    # read test data
    n_test = int(input())
    # n_test = int(input_file.readline())
    test_data = [set() for _ in range(n_test)]
    for i in range(n_test):
        # row = list(input_file.readline().split())
        row = list(input().split())
        row.pop(0)
        for word in row:
            test_data[i].add(word)

    # count up amount of classes the word refers to
    for i in range(n_train):
        words_set = train_data[i]
        current_class = train_classes[i]
        for word in words_set:
            if word in class_amount_with_word:
                class_amount_with_word[word][current_class] += 1
            else:
                class_amount_with_word[word] = [0 for _ in classes]
                class_amount_with_word[word][current_class] += 1

    # count up P(W_i|C_y)
    prob_w_c = [{} for _ in classes]
    for word in class_amount_with_word.keys():
        for c in classes:
            count = class_amount_with_word[word][c]
            prob_w_c[c][word] = float(count + alpha) / float(amount_of_class[c] + alpha * Q_dim)

    # count up P(C)
    prob_c = []
    for c in classes:
        prob_c.append(amount_of_class[c] / n_train)
        # print(prob_c[c])

    # validate calculated probability with cf sample
    # for c in classes:
    #     print(prob_w_c[c]["ant"], prob_w_c[c]["bird"], prob_w_c[c]["dog"], prob_w_c[c]["emu"], prob_w_c[c]["fish"])
    # return

    # print answer for every query X
    for i in range(n_test):
        class_score = [0.0 for _ in classes]
        for c in classes:
            if amount_of_class[c] != 0:
                # replace the probabilities product by the sum of the logarithms to avoid too small numbers
                # which would result from the probabilities product
                cur_score = _log(lmbd[c]) + _log(prob_c[c])
                for word in all_words:
                    if word in test_data[i]:
                        cur_score += _log(prob_w_c[c][word])
                    else:
                        cur_score += _log(1 - prob_w_c[c][word])
                class_score[c] = cur_score
        max_log = max(class_score)
        sum_score = 0.0
        for c in range(len(class_score)):
            if abs(class_score[c]) > eps:
                class_score[c] = _exp(class_score[c] - max_log)
                sum_score += class_score[c]
            else:
                class_score[c] = 0
        for c in classes:
            print(class_score[c] / sum_score, end=' ')
        print()


if __name__ == '__main__':
    main()
