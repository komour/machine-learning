n, m, k = map(int, input().split())
classes = [[]]
final = [[]]

for _ in range(m - 1):
    classes.append([])

for _ in range(k - 1):
    final.append([])

data = list(map(int, input().split()))

for i in range(len(data)):
    cur = data[i]
    classes[cur - 1].append(i)

c = 0
for ci in classes:
    for obj in ci:
        final[c % k].append(obj)
        c += 1

for vec in final:
    print(len(vec), end=" ")
    vec.sort()
    for obj in vec:
        print(obj + 1, end=" ")
    print()
