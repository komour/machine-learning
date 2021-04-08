import time


def examineExample():
    pass


def takeStep():
    pass


def main():
    start_time = time.time()
    # f = open('resources/SVM/input.txt')
    n = int(input())
    # n = int(f.readline())
    threshold = 5
    b = 0
    eps = 1e-9
    exec_time = 2.87
    lambdas = [0 for _ in range(n)]
    kernel_values = [[] for _ in range(n)]
    classes = []
    for i in range(n):
        line = list(map(int, input().split()))
        # line = list(map(int, f.readline().split()))
        classes.append(line[-1])
        line.pop()
        for k in line:
            kernel_values[i].append(k)
    regularization_parameter = int(input())
    # regularization_parameter = int(f.readline())
    numChanged = 0
    examineAll = True
    while numChanged > 0 or examineAll:
        numChanged = 0
        if examineAll:
            for i in range(n):
                numChanged += examineExample()
        else:
            for i in range(n):
                if lambdas[i] == 0:
                    continue



    # print(time.time() - start_time)
    for lmbd in lambdas:
        print(lmbd if lmbd > 0 else 0)
    print(b)


if __name__ == '__main__':
    main()
