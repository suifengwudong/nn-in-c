#define NN_IMPLEMENTATION
#include "../nn.h"
#include <time.h>

#define LEN(xs) (sizeof(xs) / sizeof((xs)[0]))

// 比较两个网络就地梯度 (grad_weights/grad_biases) 的最大绝对差
static nn_real max_grad_diff_NN(NN a, NN b) {
    nn_real m = 0.0f;
    for (size_t l = 0; l < a.num_layers; l++) {
        for (size_t i = 0; i < a.layers[l].grad_weights.rows; i++)
            for (size_t j = 0; j < a.layers[l].grad_weights.cols; j++) {
                nn_real d = mat_at(a.layers[l].grad_weights, i, j) - mat_at(b.layers[l].grad_weights, i, j);
                if (d < 0) d = -d;
                if (d > m) m = d;
            }
        for (size_t j = 0; j < a.layers[l].grad_biases.cols; j++) {
            nn_real d = mat_at(a.layers[l].grad_biases, 0, j) - mat_at(b.layers[l].grad_biases, 0, j);
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

    NN nn = nn_alloc(LEN(arch) - 1, arch, acts, LOSS_MSE);
    nn_rand(nn, -1.0, 1.0);

    // backprop: 就地梯度写入 nn.layers[].grad_*
    nn_backprop(nn, train_in, train_out);

    // 快照 backprop 梯度
    NN snap = nn_alloc(LEN(arch) - 1, arch, acts, LOSS_MSE);
    nn_copy(snap, nn);

    // 有限差分: 就地覆盖 nn.layers[].grad_*
    nn_finite_diff(nn, train_in, train_out, 1e-3);

    nn_real d1f = max_grad_diff_NN(snap, nn);
    printf("max|backprop - finitediff | = %.3e\n", d1f);

    // 训练,验证可收敛
    NN nn2 = nn_alloc(LEN(arch) - 1, arch, acts, LOSS_MSE);
    nn_rand(nn2, -1.0, 1.0);
    nn_real lr = 1e-1;
    for (int epoch = 0; epoch < 20000; epoch++) {
        nn_backprop(nn2, train_in, train_out);
        nn_learn(nn2, lr);
    }
    printf("final cost: %f\n", nn_cost(nn2, train_in, train_out));

    nn_free(nn); nn_free(snap); nn_free(nn2);
    mat_free(train_data);
    return d1f < 1e-3 ? 0 : 1;
}
