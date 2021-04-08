def main():
    f = open('resources/DT/input.txt')
    # features, classes, max_height = map(int, input().split())
    features, classes, max_height = map(int, f.readline().split())
    # n = int(input())
    n = int(f.readline())
    X = []
    y = []
    print(features, classes, max_height)
    print(n)
    for i in range(n):
        # line = list(map(int, input()))
        line = list(map(int, f.readline().split()))
        y.append(line.pop())
        X.append(line)


if __name__ == '__main__':
    main()
