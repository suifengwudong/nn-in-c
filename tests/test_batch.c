#define NN_IMPLEMENTATION
#include "../nn.h"
#include <time.h>

#define LEN(xs) (sizeof(xs) / sizeof((xs)[0]))

int main() {
    srand(42);

    // XOR 数据: 2 输入, 2 输出 one-hot (分类) — 测试 CE + 批量
    nn_real data_ce[] = {
        0, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 0, 1,
        1, 1, 1, 0,
    };
    Matrix td = mat_from_array(4, 4, data_ce);
    Matrix in = mat_submatrix(td, 0, 4, 0, 2);
    Matrix y = mat_submatrix(td, 0, 4, 2, 4);

    size_t arch[] = {2, 4, 2};
    // 隐藏层用 sigmoid, 避免死 ReLU 导致梯度全 0 的假象
    ActivationType acts[] = {ACT_SIGMOID, ACT_IDENTITY};

    // 批量版网络 (nn_batch_alloc, batch_size=4)
    NN nn_b = nn_batch_alloc(LEN(arch), arch, acts, LOSS_CROSS_ENTROPY, 4);
    nn_rand(nn_b, -0.5, 0.5);
    // 逐样本版网络 (同参数)
    NN nn_s = nn_alloc(LEN(arch), arch, acts, LOSS_CROSS_ENTROPY);
    nn_copy(nn_s, nn_b);

    // 批量版训练
    for (int e = 0; e < 3000; e++)
        nn_train_batch(&nn_b, in, y, 0.1f);
    // 逐样本版训练 (每步 = backprop + learn)
    for (int e = 0; e < 3000; e++) {
        nn_backprop(nn_s, in, y);
        nn_learn(nn_s, 0.1f);
    }

    // 对比两版最终输出 (用逐样本 forward 推理)
    Matrix out_b = mat_alloc(4, 2);
    Matrix out_s = mat_alloc(4, 2);
    for (size_t i = 0; i < 4; i++) {
        mat_copy(nn_b.input, mat_row(in, i));
        nn_forward(nn_b);
        mat_copy(mat_row(out_b, i), nn_b.layers[nn_b.num_fclayers - 1].activations);
        mat_copy(nn_s.input, mat_row(in, i));
        nn_forward(nn_s);
        mat_copy(mat_row(out_s, i), nn_s.layers[nn_s.num_fclayers - 1].activations);
    }
    nn_real diff = 0.0f;
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 2; j++) {
            nn_real d = mat_at(out_b, i, j) - mat_at(out_s, i, j);
            if (d < 0) d = -d;
            if (d > diff) diff = d;
        }
    printf("batch trained cost: %f\n", loss_forward(&nn_b.loss, out_b, y));
    printf("sample trained cost: %f\n", loss_forward(&nn_s.loss, out_s, y));
    printf("max|batch_out - sample_out| = %.3e\n", diff);

    mat_free(td);
    mat_free(out_b);
    mat_free(out_s);
    nn_free(nn_b);
    nn_free(nn_s);
    return diff < 1e-3 ? 0 : 1;
}
