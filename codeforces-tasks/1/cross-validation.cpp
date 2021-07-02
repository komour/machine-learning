#define us unsigned short

#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::cin.tie(nullptr);
    std::ios_base::sync_with_stdio(false);
    std::cout.precision(10);

    size_t n; // amount of objects
    size_t m; // amount of classes
    size_t k; // amount of parts

    std::cin >> n >> m >> k;

    std::vector<std::vector<size_t>> classes(m);
    std::vector<std::vector<size_t>> final(k);

    size_t cur;
    for (size_t i = 0; i < n; ++i) {
        std::cin >> cur;
        classes[cur - 1].push_back(i);
    }

    for (size_t i = 0, c = 0; i < m; ++i) {
        for (size_t j = 0; j < classes[i].size(); ++j, c++) {
            final[c % k].push_back(classes[i][j]);
//            final[c % k].push_back(i);
        }
    }

    for (auto vec : final) {
        std::cout << vec.size();
        std::sort(vec.begin(), vec.end());
        for (auto obj : vec) {
            std::cout << ' ' << ++obj;
        }
        std::cout << std::endl;
    }

    return 0;
}