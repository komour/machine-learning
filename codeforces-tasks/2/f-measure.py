def f1(prec, recall):
    return 2 * prec * recall / (prec + recall)


size = int(input())
elements = [list(map(int, input().split())) for x in range(size)]

row = [sum(i) for i in elements]
column = [sum(map(lambda x: x[i], elements)) for i in range(size)]
full = sum(row)

prec_sum = 0
recall_sum = 0
score = 0

for i in range(size):
    local_precision = 0 if row[i] == 0 else elements[i][i] / row[i]
    local_recall = 0 if column[i] == 0 else elements[i][i] / column[i]
    weight = row[i]

    prec_sum += local_precision * weight
    recall_sum += local_recall * weight
    score += f1(local_precision, local_recall) * weight

# macro
print(f1(prec_sum, recall_sum) / full)
# micro
print(score / full)
