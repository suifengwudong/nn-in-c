// MNIST 数据集神经网络训练示例, 采用小批量随机梯度下降 (SGD)
//
// 用法: mnist_nn [epochs] [batch_size] [learning_rate]
// 默认: mnist_nn 5 128 0.1
// 输出: 每 epoch 的 loss 与训练准确率写入 build/mnist_loss.csv

#define NN_IMPLEMENTATION
#define MNIST_IMPLEMENTATION
#include "../nn.h"
#include "./mnist.h"

#include <stdlib.h>
#include <time.h>

#define LEN(xs) (sizeof(xs) / sizeof((xs)[0]))
#define __FILENAME__ "mnist_nn"

// 取矩阵某一行的最大值所在列下标 (argmax)
static size_t argmax_row(const Matrix m, size_t row) {
    size_t best = 0;
    nn_real best_val = mat_at(m, row, 0);
    for (size_t j = 1; j < m.cols; j++) {
        nn_real v = mat_at(m, row, j);
        if (v > best_val) { best_val = v; best = j; }
    }
    return best;
}

// 在整个数据集上计算分类准确率
static double accuracy(NN nn, const MNIST_Data* data) {
    size_t correct = 0;
    Matrix out = nn.layers[nn.num_fclayers - 1].activations;
    for (size_t n = 0; n < data->count; n++) {
        mat_copy(nn.input, mat_row(data->images, n));
        nn_forward(nn);
        if (argmax_row(out, 0) == argmax_row(data->labels, n))
            correct++;
    }
    return (double)correct / (double)data->count;
}

int main(int argc, char** argv) {
    srand(42);

    int epochs = argc > 1 ? atoi(argv[1]) : 5;
    int batch_size = argc > 2 ? atoi(argv[2]) : 128;
    nn_real lr = argc > 3 ? (nn_real)atof(argv[3]) : 0.1f;

    MNIST_Data train = mnist_load(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte");
    MNIST_Data test = mnist_load(
        "data/mnist/t10k-images-idx3-ubyte",
        "data/mnist/t10k-labels-idx1-ubyte");

    printf("训练集: %zu 张, 测试集: %zu 张\n", train.count, test.count);

    // 网络结构: 784 -> 128 -> 64 -> 10, 输出层 identity (logits), 损失用交叉熵
    size_t arch[] = {MNIST_IMAGE_PIXELS, 128, 64, MNIST_NUM_CLASSES};
    ActivationType acts[] = {ACT_GELU, ACT_GELU, ACT_IDENTITY};
#ifdef NN_BATCH
    // 批量版: nn_batch_alloc 一次分配批量缓冲 (batch_size 为最大批次行数)
    NN nn = nn_batch_alloc(LEN(arch), arch, acts, LOSS_CROSS_ENTROPY, batch_size);
#else
    NN nn = nn_alloc(LEN(arch), arch, acts, LOSS_CROSS_ENTROPY);
#endif
    nn_rand(nn, -0.1f, 0.1f);

    // 洗牌索引 + 小批量缓冲区
    size_t* perm = (size_t*)malloc(sizeof(size_t) * train.count);
    NN_ASSERT(perm != NULL);
    for (size_t i = 0; i < train.count; i++) perm[i] = i;

    Matrix batch_in = mat_alloc(batch_size, MNIST_IMAGE_PIXELS);
    Matrix batch_out = mat_alloc(batch_size, MNIST_NUM_CLASSES);

    char file_name[256];
    snprintf(file_name, sizeof(file_name), "build/%s_%d_%d_%f_loss.csv", __FILENAME__, epochs, batch_size, lr);
    FILE* log = fopen(file_name, "w");
    NN_ASSERT(log != NULL);
    fprintf(log, "Epoch,Loss,TrainAcc\n");

    size_t num_batches = (train.count + (size_t)batch_size - 1) / (size_t)batch_size;
    time_t t0 = time(NULL), start_time = t0, end_time;
    for (int epoch = 0; epoch < epochs; epoch++) {
        start_time = time(NULL);
        // Fisher-Yates 洗牌, 保证每个 epoch 的小批量划分不同
        for (size_t i = train.count - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            size_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }

        for (size_t b = 0; b < num_batches; b++) {
            size_t start = b * (size_t)batch_size;
            size_t cnt = start + (size_t)batch_size <= train.count
                       ? (size_t)batch_size : train.count - start;
            Matrix in = mat_submatrix(batch_in, 0, cnt, 0, MNIST_IMAGE_PIXELS);
            Matrix out = mat_submatrix(batch_out, 0, cnt, 0, MNIST_NUM_CLASSES);
            for (size_t i = 0; i < cnt; i++) {
                mat_copy(mat_row(in, i), mat_row(train.images, perm[start + i]));
                mat_copy(mat_row(out, i), mat_row(train.labels, perm[start + i]));
            }
#ifdef NN_BATCH
            // 批量版: 内部 self-forward + 反向 + 更新, 一步到位
            nn_train_batch(&nn, in, out, lr);
#else
            nn_backprop(nn, in, out);
            nn_learn(nn, lr);
#endif
        }

        nn_real loss = nn_cost(nn, train.images, train.labels);
        double acc = accuracy(nn, &train);
        end_time = time(NULL);
        fprintf(log, "%d,%f,%f\n", epoch, loss, acc);
        printf("Epoch %d: loss=%f train_acc=%.4f time=%lds\n", epoch, loss, acc, (long)(end_time - start_time));
    }

    double test_acc = accuracy(nn, &test);
    printf("\n测试集准确率: %.4f 总时间 %lds\n", test_acc, (long)(time(NULL) - t0));

    fclose(log);
    free(perm);
    mat_free(batch_in);
    mat_free(batch_out);
    nn_free(nn);
    mnist_free(train);
    mnist_free(test);
    return 0;
}


