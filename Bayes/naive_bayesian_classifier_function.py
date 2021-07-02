import math


def naive_bayes_classifier(lambdas, alpha, train_input, test_input, class_amount=2):
    Q_dim = 2
    eps = 1e-5
    all_words = set()
    lmbd = lambdas
    n_train = len(train_input)
    train_classes = []
    classes = [i for i in range(class_amount)]
    train_data = [set() for _ in range(n_train)]
    class_amount_with_word = {}
    amount_of_class = [0 for _ in classes]

    # read train data
    for i in range(n_train):
        row = list(train_input[i].split())
        # row = train_input[i]
        cur_class = int(row.pop(0)) - 1
        train_classes.append(cur_class)
        amount_of_class[cur_class] += 1
        row.pop(0)
        for word in row:
            train_data[i].add(word)
            all_words.add(word)

    # read test data
    n_test = len(test_input)
    test_data = [set() for _ in range(n_test)]
    for i in range(n_test):
        row = list(test_input[i].split())
        # row = test_input[i]
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

    # print answer for every query X
    res = []
    for i in range(n_test):
        class_score = [0.0 for _ in classes]
        for c in classes:
            if amount_of_class[c] != 0:
                # replace the probabilities product by the sum of the logarithms to avoid too small numbers
                # which would result from the probabilities product
                cur_score = math.log(lmbd[c]) + math.log(prob_c[c])
                for word in all_words:
                    if word in test_data[i]:
                        cur_score += math.log(prob_w_c[c][word])
                    else:
                        cur_score += math.log(1 - prob_w_c[c][word])
                class_score[c] = cur_score
        max_log = max(class_score)
        sum_score = 0.0
        for c in range(len(class_score)):
            if abs(class_score[c]) > eps:
                class_score[c] = math.exp(class_score[c] - max_log)
                sum_score += class_score[c]
            else:
                class_score[c] = 0
        ans = []
        for c in classes:
            # print(class_score[c] / sum_score, end=' ')
            ans.append(class_score[c] / sum_score)
        res.append(ans)
    return res


def main():
    class_amount = 3
    lambdas = [1, 1, 1]
    alpha = 1
    train_input = ["1 2 ant emu",
                   "2 3 dog fish dog",
                   "3 3 bird emu ant",
                   "1 3 ant dog bird"]
    test_input = ["2 emu emu",
                  "5 emu dog fish dog fish",
                  "5 fish emu ant cat cat",
                  "2 emu cat",
                  "1 cat"]
    res = naive_bayes_classifier(lambdas, alpha, train_input, test_input, class_amount)
    for lit in res:
        for e in lit:
            print(e, end=' ')
        print()


if __name__ == '__main__':
    main()
