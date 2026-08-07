#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

#define LEN(xs) (sizeof(xs) / sizeof((xs)[0]))

// 比较 nn_backprop 与 nn_backprop_mat 的梯度,并与有限差分对照
static nn_real max_diff_NN(NN a, NN b) {
    nn_real m = 0.0f;
    for (size_t l = 0; l < a.num_layers - 1; l++) {
        for (size_t i = 0; i < a.weights[l].rows; i++)
            for (size_t j = 0; j < a.weights[l].cols; j++) {
                nn_real d = mat_at(a.weights[l], i, j) - mat_at(b.weights[l], i, j);
                if (d < 0) d = -d;
                if (d > m) m = d;
            }
        for (size_t j = 0; j < a.biases[l].cols; j++) {
            nn_real d = mat_at(a.biases[l], 0, j) - mat_at(b.biases[l], 0, j);
            if (d < 0) d = -d;
            if (d > m) m = d;
        }
    }
    return m;
}

int main() {
    srand(42);  // 固定种子,保证梯度对比与训练结果可复现

    nn_real data[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
        0.5f, 0.5f, 1,
    };
    Matrix train_data = mat_from_array(5, 3, data);
    Matrix train_in = mat_submatrix(train_data, 0, 5, 0, 2);
    Matrix train_out = mat_submatrix(train_data, 0, 5, 2, 3);

    size_t arch[] = {2, 3, 2, 1};
    ActivationType acts[] = {ACT_SIGMOID, ACT_RELU, ACT_SIGMOID};

    NN nn = nn_alloc(LEN(arch), arch, acts);
    nn_rand(nn, -1.0, 1.0);

    NN g1 = nn_alloc_grad(LEN(arch), arch);   // 标量版梯度
    NN g2 = nn_alloc_grad(LEN(arch), arch);   // 矩阵版梯度
    NN gf = nn_alloc_grad(LEN(arch), arch);   // 有限差分梯度

    nn_backprop(nn, g1, train_in, train_out);
    nn_backprop_mat(nn, g2, train_in, train_out);
    nn_finite_diff(nn, gf, train_in, train_out, 1e-3);

    nn_real d12 = max_diff_NN(g1, g2);
    nn_real d1f = max_diff_NN(g1, gf);
    nn_real d2f = max_diff_NN(g2, gf);

    printf("max|backprop - backprop_mat| = %.3e\n", d12);
    printf("max|backprop - finitediff | = %.3e\n", d1f);
    printf("max|backprop_mat - finitediff| = %.3e\n", d2f);

    // 用矩阵版训练,验证可收敛
    NN nn2 = nn_alloc(LEN(arch), arch, acts);
    nn_rand(nn2, -1.0, 1.0);
    nn_real lr = 1e-1;
    for (int epoch = 0; epoch < 20000; epoch++) {
        nn_backprop_mat(nn2, g2, train_in, train_out);
        nn_learn(nn2, g2, lr);
    }
    printf("mat-version final cost: %f\n", nn_cost(nn2, train_in, train_out));

    nn_free(nn); nn_free(g1); nn_free(g2); nn_free(gf); nn_free(nn2);
    mat_free(train_data);
    return d12 < 1e-6 ? 0 : 1;
}
