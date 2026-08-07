#define NN_IMPLEMENTATION
#include "nn.h"

#include <time.h>
#define LEN(xs) (sizeof(xs) / sizeof((xs)[0]))

int main() {
    srand(time(0));

    nn_real data[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0
    };
    Matrix train_data = mat_from_array(4, 3, data);
    Matrix train_in = mat_submatrix(train_data, 0, 4, 0, 2);
    Matrix train_out = mat_submatrix(train_data, 0, 4, 2, 3);

    size_t arch[] = {2, 2, 1};
    ActivationType acts[] = {ACT_SIGMOID, ACT_SIGMOID};
    NN nn = nn_alloc(LEN(arch), arch, acts);
    nn_rand(nn, 0, 1.0);
    NN g = nn_alloc_grad(LEN(arch), arch);

    printf("Cost: %f\n", nn_cost(nn, train_in, train_out));

    nn_real lr = 1e-1;
    for (int epoch = 0; epoch < 20000; epoch++) {
        nn_backprop(nn, g, train_in, train_out);
        nn_learn(nn, g, lr);
    }

    printf("Cost: %f\n", nn_cost(nn, train_in, train_out));

    // evaluate
    for (size_t n = 0; n < 4; n++) {
        mat_copy(nn.activations[0], mat_row(train_in, n));
        nn_forward(nn);
        printf("(%f, %f) -> %f (expected %f)\n",
            mat_at(train_in, n, 0), mat_at(train_in, n, 1),
            mat_at(nn.activations[2], 0, 0), mat_at(train_out, n, 0));
    }

    mat_free(train_data);
    nn_free(nn);
    nn_free(g);
    return 0;
}
