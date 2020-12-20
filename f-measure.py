def f1(p, r):
    return 2 * p * r / (p + r)


k = int(input())
row_sum = [0] * k
column_sum = [0] * k
true_positive = [0.0] * k
amount = 0

for i in range(k):
    row = list(map(int, input().split()))
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

print(macro)
print(micro)
