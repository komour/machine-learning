#define us unsigned short

#include <iostream>
#include <vector>

double f1(double prec, double recall) {
    return 2 * prec * recall / (prec + recall);
}

int main() {
    std::cin.tie(nullptr);
    std::ios_base::sync_with_stdio(false);
    std::cout.precision(10);

    us k;
    std::cin >> k;

    std::vector<us> row_sum(k, 0);
    std::vector<us> column_sum(k, 0);
    std::vector<us> true_positive(k, 0);
    us amount = 0;
    us x;
    for (us i = 0; i < k; ++i) {
        for (us j = 0; j < k; ++j) {
            std::cin >> x;
            amount += x;
            row_sum[i] += x;
            column_sum[j] += x;
            if (i == j) {
                true_positive[i] = x;
            }
        }
    }

    std::vector<double> recall(k, 0), prec(k, 0);
    double recall_sum = 0;
    double prec_sum = 0;
    double micro_sum = 0;

    for (us i = 0; i < k; ++i) {
        if (true_positive[i] == 0) continue;
        prec[i] = double(true_positive[i]) / row_sum[i];
        prec_sum += prec[i] * row_sum[i];

        recall[i] = double(true_positive[i]) / column_sum[i];
        recall_sum += recall[i] * row_sum[i];

        micro_sum += f1(recall[i], prec[i]) * row_sum[i];
    }

    double prec_average = prec_sum / amount;
    double recall_average = recall_sum / amount;

    double macro = f1(prec_average, recall_average);
    double micro = micro_sum / amount;

    std::cout << macro << '\n' << micro;

    return 0;
}