// stb 风格神经网络库实现

#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>

#define nn_real float

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#define NN_EPSILON 1e-8f

// ==================== Activation System ====================

static inline nn_real identityf(nn_real x) {
    return x;
}

static inline nn_real sigmoidf(nn_real x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline nn_real reluf(nn_real x) {
    return x > 0.0f ? x : 0.0f;
}

static inline nn_real geluf(nn_real x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

static inline nn_real identity_gradf(nn_real x) {
    (void)x;
    return 1.0f;
}

static inline nn_real sigmoid_gradf(nn_real x) {
    nn_real s = sigmoidf(x);
    return s * (1.0f - s);
}

static inline nn_real relu_gradf(nn_real x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

static inline nn_real gelu_gradf(nn_real x) {
    nn_real c = sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f));
    nn_real t = tanhf(c);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * sqrtf(2.0f / M_PI) * (1.0f + 0.134145f * x * x);
}

#define ACTIVATIONS \
    X(ACT_IDENTITY, identityf,  identity_gradf) \
    X(ACT_SIGMOID,  sigmoidf,   sigmoid_gradf)  \
    X(ACT_RELU,     reluf,      relu_gradf)     \
    X(ACT_GELU,     geluf,      gelu_gradf)

#define X(id, fn, grad_fn) id,
typedef enum { ACTIVATIONS } ActivationType;
#undef X

static inline nn_real activate(ActivationType type, nn_real x) {
    switch (type) {
        #define X(id, fn, grad_fn) case id: return fn(x);
        ACTIVATIONS
        #undef X
    }
    return x;
}

static inline nn_real activate_grad(ActivationType type, nn_real x) {
    switch (type) {
        #define X(id, fn, grad_fn) case id: return grad_fn(x);
        ACTIVATIONS
        #undef X
    }
    return 0.0f;
}

// ==================== Matrix ====================

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;   // 相邻两行首元素之间的元素间隔 (列方向恒连续, 便于 SIMD)
    nn_real *elements;
} Matrix;

#define mat_at(m, r, c) ((m).elements[(r) * (m).stride + (c)])

Matrix mat_alloc(size_t rows, size_t cols);
void mat_free(Matrix m);
Matrix mat_from_array(size_t rows, size_t cols, const nn_real* array);
void mat_fill(Matrix m, nn_real value);
void mat_rand(Matrix m, nn_real min, nn_real max);
Matrix mat_submatrix(Matrix m, size_t row_start, size_t row_end, size_t col_start, size_t col_end);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dest, const Matrix src);

// ==================== BLAS api ====================
// dst = beta·dst + alpha·(a'·b), a' 由 trans_a 决定是否取转置
// b 恒以行主序、内层累加轴连续为前置条件 (需转置的一方预存成转置布局, 如 Layer.weights_T)
void mat_gemm(Matrix dst, const Matrix a, int trans_a, const Matrix b, nn_real alpha, nn_real beta);
void mat_transpose_into(Matrix dst, const Matrix src);
// dst = beta·dst + alpha·a
void mat_axpy(Matrix dst, const Matrix a, nn_real alpha, nn_real beta);

// aliases
#define mat_mul(dst, a, b)        mat_gemm(dst, a, 0, b, 1.0f, 0.0f)
#define mat_mul_acc(dst, a, b, s) mat_gemm(dst, a, 0, b, s, 1.0f)
#define mat_add(dst, a)           mat_axpy(dst, a, 1.0f, 1.0f)
#define mat_add_scaled(dst, a, s) mat_axpy(dst, a, s, 1.0f)
#define mat_scale(dst, a, s)      mat_axpy(dst, a, s, 0.0f)

void _mat_print(const Matrix m, const char* name);
#define mat_print(m) _mat_print(m, #m)

// ==================== Layer ====================

typedef struct {
    size_t input_size;
    size_t output_size;
    Matrix weights;    // output×input (W): 用户/标量路径/保存/加载用
    Matrix weights_T;  // input×output (Wᵀ): batch 前向 SIMD 用, 恒等于 weights 的转置
    Matrix biases;
    Matrix weighted_sum;  // z = W·x + b
    Matrix activations;   // a = sigma(z)
    ActivationType act;

    Matrix grad_weights;
    Matrix grad_biases;
} Layer;

Layer layer_alloc(size_t output_size, size_t input_size, ActivationType act);
void layer_free(Layer layer);
void layer_forward(Layer* layer, const Matrix input);
void layer_activate(Layer* layer);
void layer_pass(Layer* layer, const Matrix input);
void layer_backward(Layer* layer, Matrix input, const Matrix delta);

// ==================== LossLayer ====================
// 损失层: 无权重, 只计算 loss 值与对输入的梯度 (通常是 logits)
// api: forward & backward; distributed by LossType

#define LOSS_LAYERS \
    X(LOSS_MSE,           mse_layer_forward,  mse_layer_backward) \
    X(LOSS_CROSS_ENTROPY, ce_layer_forward,   ce_layer_backward)

#define X(id, fwd, bwd) id,
typedef enum { LOSS_LAYERS } LossType;
#undef X

typedef struct {
    LossType type;
    Matrix probs;
} LossLayer;

nn_real mse_layer_forward(LossLayer* layer, const Matrix in, const Matrix y);
void mse_layer_backward(LossLayer* layer, Matrix delta, const Matrix in, const Matrix y);
nn_real ce_layer_forward(LossLayer* layer, const Matrix in, const Matrix y);
void ce_layer_backward(LossLayer* layer, Matrix delta, const Matrix in, const Matrix y);

nn_real loss_forward(LossLayer* layer, const Matrix in, const Matrix y);
void loss_backward(LossLayer* layer, Matrix delta, const Matrix in, const Matrix y);

// ==================== Neural Network ====================

typedef struct {
    size_t num_fclayers; // full connected layers
    Matrix input;        // 1×arch[0]:  temps
    Layer *layers;
    LossLayer loss;      // single loss layer, forward/backward

    Matrix *batch_a;     // num_fclayers 个: batch×arch[i+1] 每层激活
    Matrix *batch_z;     // num_fclayers 个: batch×arch[i+1] pre-activation
    Matrix *batch_d;     // num_fclayers 个: batch×arch[i+1] delta
} NN;

NN nn_alloc(size_t num_layers, const size_t* layer_sizes, const ActivationType* acts, LossType loss);
void nn_free(NN nn);
void nn_forward(NN nn);
void nn_rand(NN nn, nn_real min, nn_real max);
void nn_zero(NN nn);
void nn_copy(NN dest, NN src);
nn_real nn_cost(NN nn, Matrix train_in, Matrix train_out);
void nn_finite_diff(NN nn, Matrix train_in, Matrix train_out, nn_real eps);
void nn_backprop(NN nn, Matrix train_in, Matrix train_out);
void nn_learn(NN nn, nn_real learning_rate);

// ==================== NN_BATCH ====================

NN nn_batch_alloc(size_t num_layers, const size_t* layer_sizes, const ActivationType* acts, LossType loss, size_t batch_size);
// 批量训练
void nn_train_batch(NN* nn, Matrix in, Matrix y, nn_real lr);


// ==================== Implementation ====================

#ifdef NN_IMPLEMENTATION

Matrix mat_alloc(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.elements = (nn_real*)NN_MALLOC(sizeof(*m.elements) * rows * cols);
    NN_ASSERT(m.elements != NULL);
    return m;
}

Matrix mat_from_array(size_t rows, size_t cols, const nn_real* array) {
    Matrix m = mat_alloc(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            mat_at(m, i, j) = array[i * cols + j];
    return m;
}

void mat_free(Matrix m) {
    free(m.elements);
}

void mat_fill(Matrix m, nn_real value) {
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            mat_at(m, i, j) = value;
}

void mat_rand(Matrix m, nn_real min, nn_real max) {
    NN_ASSERT(min < max);
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            mat_at(m, i, j) = min + (max - min) * ((nn_real)rand() / RAND_MAX);
}

Matrix mat_submatrix(Matrix m, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    NN_ASSERT(row_start < row_end && row_end <= m.rows);
    NN_ASSERT(col_start < col_end && col_end <= m.cols);
    Matrix sub;
    sub.rows = row_end - row_start;
    sub.cols = col_end - col_start;
    sub.stride = m.stride;
    sub.elements = &mat_at(m, row_start, col_start);
    return sub;
}

Matrix mat_row(Matrix m, size_t row) {
    NN_ASSERT(row < m.rows);
    return (Matrix){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .elements = &mat_at(m, row, 0),
    };
}

void mat_copy(Matrix dest, const Matrix src) {
    NN_ASSERT(dest.rows == src.rows && dest.cols == src.cols);
    for (size_t i = 0; i < src.rows; i++)
        for (size_t j = 0; j < src.cols; j++)
            mat_at(dest, i, j) = mat_at(src, i, j);
}

// dst = beta·dst + alpha·(a'·b), a' 由 trans_a 决定是否取转置
// 内层一律沿 j 累加 (i-k-j): b[k][j] 行主序连续, a_ik 为标量复用, 即默认 SIMD 顺序
// 需要 b 转置时提前实体化 (如 Layer.weights_T), 无需运行时标志
void mat_gemm(Matrix dst, const Matrix a, int trans_a, const Matrix b, nn_real alpha, nn_real beta) {
    size_t M = trans_a ? a.cols : a.rows;   // a' 行数
    size_t K = trans_a ? a.rows : a.cols;   // 内层累加维
    size_t N = b.cols;                      // b 列数
    NN_ASSERT(K == b.rows);
    NN_ASSERT(dst.rows == M && dst.cols == N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++)
            mat_at(dst, i, j) *= beta;
        for (size_t k = 0; k < K; k++) {
            nn_real a_ik = trans_a ? mat_at(a, k, i) : mat_at(a, i, k);
            #pragma GCC ivdep
            for (size_t j = 0; j < N; j++)
                mat_at(dst, i, j) += alpha * a_ik * mat_at(b, k, j);
        }
    }
}

// dst = src^T (dst.rows==src.cols && dst.cols==src.rows)
void mat_transpose_into(Matrix dst, const Matrix src) {
    NN_ASSERT(dst.rows == src.cols && dst.cols == src.rows);
    for (size_t i = 0; i < src.rows; i++)
        for (size_t j = 0; j < src.cols; j++)
            mat_at(dst, j, i) = mat_at(src, i, j);
}

void mat_axpy(Matrix dst, const Matrix a, nn_real alpha, nn_real beta) {
    NN_ASSERT(dst.rows == a.rows && dst.cols == a.cols);
    for (size_t i = 0; i < a.rows; i++)
        for (size_t j = 0; j < a.cols; j++)
            mat_at(dst, i, j) = beta * mat_at(dst, i, j) + alpha * mat_at(a, i, j);
}

void _mat_print(const Matrix m, const char* name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++)
            printf("%f\t", mat_at(m, i, j));
        printf("\n");
    }
    printf("]\n");
}

Layer layer_alloc(size_t output_size, size_t input_size, ActivationType act) {
    Layer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.act = act;
    layer.weights = mat_alloc(output_size, input_size);
    layer.weights_T = mat_alloc(input_size, output_size);
    mat_fill(layer.weights_T, 0.0f);  // weights 未初始化, 先填 0 保持权重同步不变式
    layer.biases = mat_alloc(1, output_size);
    layer.weighted_sum = mat_alloc(1, output_size);
    layer.activations = mat_alloc(1, output_size);
    layer.grad_weights = mat_alloc(output_size, input_size);
    layer.grad_biases = mat_alloc(1, output_size);
    return layer;
}

void layer_free(Layer layer) {
    mat_free(layer.weights);
    mat_free(layer.weights_T);
    mat_free(layer.biases);
    mat_free(layer.weighted_sum);
    mat_free(layer.activations);
    mat_free(layer.grad_weights);
    mat_free(layer.grad_biases);
}

// 权重同步不变式: weights_T ≡ weights^T; 任何 weights 变更后调用此函数刷新
void layer_sync_weights_T(Layer* layer) {
    mat_transpose_into(layer->weights_T, layer->weights);
}

// z = W·x + b;  a = act(z), 同时记录 z 供反向使用
void layer_forward(Layer* layer, const Matrix input) {
    NN_ASSERT(input.rows == 1 && input.cols == layer->input_size);
    for (size_t j = 0; j < layer->output_size; j++) {
        nn_real sum = mat_at(layer->biases, 0, j);
        for (size_t k = 0; k < layer->input_size; k++)
            sum += mat_at(input, 0, k) * mat_at(layer->weights, j, k);
        mat_at(layer->weighted_sum, 0, j) = sum;
        mat_at(layer->activations, 0, j) = sum;
    }
}

void layer_activate(Layer* layer) {
    for (size_t i = 0; i < layer->output_size; i++)
        mat_at(layer->activations, 0, i) = activate(layer->act, mat_at(layer->activations, 0, i));
}

void layer_pass(Layer* layer, const Matrix input) {
    layer_forward(layer, input);
    layer_activate(layer);
}

void layer_backward(Layer* layer, Matrix input, const Matrix delta) {
    // 1. δz_j = delta_j · act'(z_j), 暂存到 activations
    for (size_t j = 0; j < layer->output_size; j++) {
        mat_at(layer->activations, 0, j) = mat_at(delta, 0, j)
            * activate_grad(layer->act, mat_at(layer->weighted_sum, 0, j));
    }

    // 2. grad_weights += δz^T · input; grad_biases += δz
    mat_gemm(layer->grad_weights, layer->activations, 1, input, 1.0f, 1.0f);
    mat_add(layer->grad_biases, layer->activations);

    // 3. 回传: input = δz · W (1×C 乘 C×K → 1×K), 就地覆盖, 不再需要 delta 字段
    mat_mul(input, layer->activations, layer->weights);
}

// ==================== LossLayer impl ====================
// 每个 LossLayer 提供两个接口: forward(算批次平均损失) / backward(给对 logits 的梯度)
// forward 可能归一化 (CE 写入 layer->probs), backward 复用该概率

// --- MSE 层: 输入即输出, 无需归一化 ---
//   ℓ = 1/n Σ_n Σ_j (a_j - y_j)²
nn_real mse_layer_forward(LossLayer* layer, const Matrix in, const Matrix y) {
    (void)layer;  // MSE 不归一化, 不需要 layer
    NN_ASSERT(in.rows == y.rows && in.cols == y.cols);
    nn_real sum = 0.0f;
    for (size_t i = 0; i < in.rows; i++) {
        for (size_t j = 0; j < in.cols; j++) {
            nn_real d = mat_at(mat_row(in, i), 0, j) - mat_at(mat_row(y, i), 0, j);
            sum += d * d;
        }
    }
    return sum / (nn_real)in.rows;
}

//  δ = 2(a - y)  (a 即 logits, 输出层 act=ACT_IDENTITY)
void mse_layer_backward(LossLayer* layer, Matrix delta, const Matrix in, const Matrix y) {
    (void)layer;
    NN_ASSERT(delta.rows == 1 && in.rows == 1 && y.rows == 1 && delta.cols == in.cols && in.cols == y.cols);
    for (size_t j = 0; j < in.cols; j++)
        mat_at(delta, 0, j) = 2.0f * (mat_at(in, 0, j) - mat_at(y, 0, j));
}

// --- CE 层: 把 logits softmax 归一化到 layer->probs, 再算交叉熵 ---
//   p = softmax(z);  ℓ = -1/n Σ_n Σ_j y_j·log(p_j)
nn_real ce_layer_forward(LossLayer* layer, const Matrix in, const Matrix y) {
    NN_ASSERT(in.rows == y.rows && in.cols == y.cols);
    NN_ASSERT(layer->probs.rows == 1 && layer->probs.cols == in.cols);
    nn_real sum = 0.0f;
    for (size_t i = 0; i < in.rows; i++) {
        nn_real max = mat_at(in, i, 0);
        for (size_t j = 1; j < in.cols; j++)
            if (mat_at(in, i, j) > max) max = mat_at(in, i, j);
        nn_real expsum = 0.0f;
        for (size_t j = 0; j < in.cols; j++) {
            nn_real e = expf(mat_at(in, i, j) - max);
            mat_at(layer->probs, 0, j) = e;
            expsum += e;
        }
        for (size_t j = 0; j < in.cols; j++)
            mat_at(layer->probs, 0, j) /= expsum;
        // 累加该行 CE: -Σ y_j·log(p_j)
        for (size_t j = 0; j < in.cols; j++)
            sum += mat_at(y, i, j) * logf(mat_at(layer->probs, 0, j) + NN_EPSILON);
    }
    return -sum / (nn_real)in.rows;
}

//  δ = (p - y)  (softmax+CE 配对后对 logits 的梯度, σ' 抵消; p 来自 forward 的 probs)
void ce_layer_backward(LossLayer* layer, Matrix delta, const Matrix in, const Matrix y) {
    (void)in;  // 梯度直接用 forward 保存的 probs, 不需要 logits
    NN_ASSERT(delta.rows == 1 && y.rows == 1 && delta.cols == y.cols);
    NN_ASSERT(layer->probs.rows == 1 && layer->probs.cols == y.cols);
    Matrix p = mat_row(layer->probs, 0);
    Matrix t = mat_row(y, 0);
    for (size_t j = 0; j < y.cols; j++)
        mat_at(delta, 0, j) = mat_at(p, 0, j) - mat_at(t, 0, j);
}

// --- 统一分发器 (由 LOSS_LAYERS X-macro 生成) ---

nn_real loss_forward(LossLayer* layer, const Matrix in, const Matrix y) {
    switch (layer->type) {
        #define X(id, fwd, bwd) case id: return fwd(layer, in, y);
        LOSS_LAYERS
        #undef X
    }
    return 0.0f;
}

void loss_backward(LossLayer* layer, Matrix delta, const Matrix in, const Matrix y) {
    switch (layer->type) {
        #define X(id, fwd, bwd) case id: bwd(layer, delta, in, y); break;
        LOSS_LAYERS
        #undef X
    }
}

// ==================== NN impl ====================

// {1, 2, 1} = {input_layer, hidden_layers, output_layer}
// 语义: num_layers = 层总数 (含输入层), 即 arch 数组长度 LEN(arch)
//       fclayers = 全连接层数, 即 num_layers-1 个
NN nn_alloc(size_t num_layers, const size_t* layer_sizes, const ActivationType* acts, LossType loss) {
    NN_ASSERT(num_layers >= 2);
    NN nn;
    nn.num_fclayers = num_layers - 1;  // 权重层数 (nn.layers 数组长度)
    nn.input = mat_alloc(1, layer_sizes[0]);
    nn.layers = (Layer*)NN_MALLOC(sizeof(Layer) * (num_layers - 1));
    NN_ASSERT(nn.layers != NULL);

    for (size_t i = 0; i < num_layers - 1; i++) {
        nn.layers[i] = layer_alloc(layer_sizes[i + 1], layer_sizes[i], acts[i]);
    }

    nn.loss.type = loss;
    nn.loss.probs = mat_alloc(1, layer_sizes[num_layers - 1]);
    nn.batch_a = NULL;
    nn.batch_z = NULL;
    nn.batch_d = NULL;
    return nn;
}

void nn_free(NN nn) {
    mat_free(nn.input);
    for (size_t i = 0; i < nn.num_fclayers; i++)
        layer_free(nn.layers[i]);
    free(nn.layers);
    mat_free(nn.loss.probs);
    if (nn.batch_a) {
        for (size_t i = 0; i < nn.num_fclayers; i++) {
            mat_free(nn.batch_a[i]);
            mat_free(nn.batch_z[i]);
            mat_free(nn.batch_d[i]);
        }
        free(nn.batch_a);
        free(nn.batch_z);
        free(nn.batch_d);
    }
}

// 前向: 输入在 nn.input, 逐层 layer_pass (仿射→激活)
void nn_forward(NN nn) {
    Matrix in = nn.input;
    for (size_t i = 0; i < nn.num_fclayers; i++) {
        layer_pass(&nn.layers[i], in);
        in = nn.layers[i].activations;
    }
}

void nn_rand(NN nn, nn_real min, nn_real max) {
    for (size_t i = 0; i < nn.num_fclayers; i++) {
        mat_rand(nn.layers[i].weights, min, max);
        layer_sync_weights_T(&nn.layers[i]);
        mat_rand(nn.layers[i].biases, min, max);
    }
}

void nn_zero(NN nn) {
    for (size_t i = 0; i < nn.num_fclayers; i++) {
        mat_fill(nn.layers[i].grad_weights, 0.0);
        mat_fill(nn.layers[i].grad_biases, 0.0);
    }
}

void nn_copy(NN dest, NN src) {
    NN_ASSERT(dest.num_fclayers == src.num_fclayers);
    dest.loss.type = src.loss.type;
    mat_copy(dest.input, src.input);
    for (size_t i = 0; i < dest.num_fclayers; i++) {
        mat_copy(dest.layers[i].weights, src.layers[i].weights);
        layer_sync_weights_T(&dest.layers[i]);
        mat_copy(dest.layers[i].biases, src.layers[i].biases);
        dest.layers[i].act = src.layers[i].act;
        mat_copy(dest.layers[i].grad_weights, src.layers[i].grad_weights);
        mat_copy(dest.layers[i].grad_biases, src.layers[i].grad_biases);
    }
    mat_copy(dest.loss.probs, src.loss.probs);
    if (src.batch_a && dest.batch_a) {
        for (size_t i = 0; i < dest.num_fclayers; i++) {
            mat_copy(dest.batch_a[i], src.batch_a[i]);
            mat_copy(dest.batch_z[i], src.batch_z[i]);
            mat_copy(dest.batch_d[i], src.batch_d[i]);
        }
    }
}

nn_real nn_cost(NN nn, Matrix train_in, Matrix train_out) {
    NN_ASSERT(train_in.rows == train_out.rows);
    nn_real sum = 0.0;
    Matrix out = nn.layers[nn.num_fclayers - 1].activations;
    for (size_t n = 0; n < train_in.rows; n++) {
        mat_copy(nn.input, mat_row(train_in, n));
        nn_forward(nn);
        sum += loss_forward(&nn.loss, out, mat_row(train_out, n));
    }
    return sum / train_in.rows;
}

// 有限差分
void nn_finite_diff(NN nn, Matrix train_in, Matrix train_out, nn_real eps) {
    nn_real original_cost = nn_cost(nn, train_in, train_out);
    nn_real saved;

    for (size_t l = 0; l < nn.num_fclayers; l++) {
        for (size_t i = 0; i < nn.layers[l].weights.rows; i++) {
            for (size_t j = 0; j < nn.layers[l].weights.cols; j++) {
                saved = mat_at(nn.layers[l].weights, i, j);
                mat_at(nn.layers[l].weights, i, j) += eps;
                mat_at(nn.layers[l].grad_weights, i, j) = (nn_cost(nn, train_in, train_out) - original_cost) / eps;
                mat_at(nn.layers[l].weights, i, j) = saved;
            }
        }
        for (size_t j = 0; j < nn.layers[l].biases.cols; j++) {
            saved = mat_at(nn.layers[l].biases, 0, j);
            mat_at(nn.layers[l].biases, 0, j) += eps;
            mat_at(nn.layers[l].grad_biases, 0, j) = (nn_cost(nn, train_in, train_out) - original_cost) / eps;
            mat_at(nn.layers[l].biases, 0, j) = saved;
        }
    }
}

/**
 * @brief 反向传播
 * @param nn 神经网络
 * @param train_in 训练输入
 * @param train_out 训练输出
 */
void nn_backprop(NN nn, Matrix train_in, Matrix train_out) {
    NN_ASSERT(train_in.rows == train_out.rows);
    nn_zero(nn);

    size_t L = nn.num_fclayers - 1;
    Matrix out = nn.layers[L].activations;

    for (size_t n = 0; n < train_in.rows; n++) {
        mat_copy(nn.input, mat_row(train_in, n));
        nn_forward(nn);

        // 输出层: loss_forward 算损失 (CE 内部归一化到 probs), loss_backward 给对 logits 的梯度
        loss_forward(&nn.loss, out, mat_row(train_out, n));
        loss_backward(&nn.loss, out, out, mat_row(train_out, n));

        // 反向: l = L ... 0; 每层就地累积梯度, 回传梯度写入 input (即上一层的输出缓冲)
        Matrix delta = out;
        for (size_t l = L; l < nn.num_fclayers; l--) {
            Matrix in = (l == 0) ? nn.input : nn.layers[l - 1].activations;
            layer_backward(&nn.layers[l], in, delta);
            delta = in;
            if (l == 0) break;
        }
    }

    // 平均: grad /= n
    nn_real inv = 1.0f / (nn_real)train_in.rows;
    for (size_t l = 0; l < nn.num_fclayers; l++) {
        mat_scale(nn.layers[l].grad_weights, nn.layers[l].grad_weights, inv);
        mat_scale(nn.layers[l].grad_biases, nn.layers[l].grad_biases, inv);
    }
}

// 用各层就地梯度更新参数
void nn_learn(NN nn, nn_real learning_rate) {
    for (size_t l = 0; l < nn.num_fclayers; l++) {
        // val -= learning_rate * grad
        mat_axpy(nn.layers[l].weights, nn.layers[l].grad_weights, -learning_rate, 1.0f);
        mat_axpy(nn.layers[l].biases, nn.layers[l].grad_biases, -learning_rate, 1.0f);
        layer_sync_weights_T(&nn.layers[l]);
    }
}

// batch_alloc
NN nn_batch_alloc(size_t num_layers, const size_t* layer_sizes, const ActivationType* acts, LossType loss, size_t batch_size) {
    NN nn = nn_alloc(num_layers, layer_sizes, acts, loss);
    nn.batch_a = (Matrix*)NN_MALLOC(sizeof(Matrix) * nn.num_fclayers);
    nn.batch_z = (Matrix*)NN_MALLOC(sizeof(Matrix) * nn.num_fclayers);
    nn.batch_d = (Matrix*)NN_MALLOC(sizeof(Matrix) * nn.num_fclayers);
    NN_ASSERT(nn.batch_a && nn.batch_z && nn.batch_d);
    for (size_t l = 0; l < nn.num_fclayers; l++) {
        size_t cols = nn.layers[l].output_size;
        nn.batch_a[l] = mat_alloc(batch_size, cols);
        nn.batch_z[l] = mat_alloc(batch_size, cols);
        nn.batch_d[l] = mat_alloc(batch_size, cols);
    }
    return nn;
}

// 批量训练一步: self-forward → 反向累积梯度 → 更新参数
//   in/y 均 batch×arch, y 为真值; 只处理前 in.rows 行 (最后一批可能不满 batch_size)
// 每层:
//   Z_l = A_{l-1}·W_l^T + b_l;  A_l = act(Z_l)          (前向)
//   dW_l += A_{l-1}^T·δ_l;  db_l += Σ_i δ_l[i][:]       (反向)
//   δ_{l-1} = δ_l·W_l ⊙ act'(z_{l-1})
void nn_train_batch(NN* nn, Matrix in, Matrix y, nn_real lr) {
    NN_ASSERT(nn->batch_a != NULL && "需要 nn_batch_alloc 分配");
    NN_ASSERT(in.rows == y.rows && in.cols == nn->layers[0].input_size && y.cols == nn->layers[nn->num_fclayers - 1].output_size);
    NN_ASSERT(in.rows <= nn->batch_a[0].rows && "batch 行数超过批量缓冲容量");
    nn_zero(*nn);
    size_t n = in.rows;
    size_t L = nn->num_fclayers - 1;

    // self-forward: 逐层写到 batch_z/a 子视图, 输出即 batch_a[L] 子视图
    Matrix out = mat_submatrix(nn->batch_a[L], 0, n, 0, nn->layers[L].output_size);
    {
        Matrix prev = in;
        for (size_t l = 0; l < nn->num_fclayers; l++) {
            Layer *lay = &nn->layers[l];
            size_t cols = lay->output_size;
            Matrix z = mat_submatrix(nn->batch_z[l], 0, n, 0, cols);
            Matrix a = mat_submatrix(nn->batch_a[l], 0, n, 0, cols);
            mat_gemm(z, prev, 0, lay->weights_T, 1.0f, 0.0f);  // A·Wᵀ (weights_T 即 Wᵀ 预转置)
            for (size_t i = 0; i < n; i++) {
                mat_add(mat_row(z, i), lay->biases);
                for (size_t j = 0; j < cols; j++)
                    mat_at(a, i, j) = activate(lay->act, mat_at(z, i, j));
            }
            prev = a;
        }
    }

    // 输出层 δ: 逐行 loss (CE: 每行 loss_forward 归一化到 probs, 再 δ = p-y)
    for (size_t i = 0; i < n; i++) {
        Matrix orow = mat_row(out, i);
        Matrix trow = mat_row(y, i);
        loss_forward(&nn->loss, orow, trow);  // 单行: CE 写入 probs (1×C)
        loss_backward(&nn->loss, orow, orow, trow);  // δ = ∂L/∂logits, 就地写 orow
        mat_copy(mat_row(nn->batch_d[L], i), orow);
    }

    // 从后往前: l = L ... 0
    for (size_t l = L; l < nn->num_fclayers; l--) {
        Layer *lay = &nn->layers[l];
        Matrix a_prev = (l == 0) ? in : mat_submatrix(nn->batch_a[l - 1], 0, n, 0, nn->layers[l - 1].output_size);
        Matrix d_l = mat_submatrix(nn->batch_d[l], 0, n, 0, lay->output_size);

        // dW_l += δ_l^T · a_prev (δ_l: n×C, a_prev: n×K → C×K, 与 grad_weights 同形状)
        mat_gemm(lay->grad_weights, d_l, 1, a_prev, 1.0f, 1.0f);

        // db_l += Σ_i δ_l[i][:]
        for (size_t i = 0; i < n; i++)
            mat_add(lay->grad_biases, mat_row(d_l, i));

        // 回传 δ_{l-1} = δ_l · W_l (n×C 乘 C×K → n×K)
        if (l > 0) {
            Layer *prevlay = &nn->layers[l - 1];
            Matrix d_prev = mat_submatrix(nn->batch_d[l - 1], 0, n, 0, prevlay->output_size);
            mat_gemm(d_prev, d_l, 0, lay->weights, 1.0f, 0.0f);
            // ⊙ act'(z_{l-1})
            Matrix z_prev = mat_submatrix(nn->batch_z[l - 1], 0, n, 0, prevlay->output_size);
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < prevlay->output_size; j++)
                    mat_at(d_prev, i, j) *=
                        activate_grad(prevlay->act, mat_at(z_prev, i, j));
        }
        if (l == 0) break;
    }

    // 平均 + 更新参数
    nn_real inv = 1.0f / (nn_real)n;
    for (size_t l = 0; l < nn->num_fclayers; l++) {
        // grad /= n
        mat_scale(nn->layers[l].grad_weights, nn->layers[l].grad_weights, inv);
        mat_scale(nn->layers[l].grad_biases, nn->layers[l].grad_biases, inv);
        // param -= lr * grad
        mat_axpy(nn->layers[l].weights, nn->layers[l].grad_weights, -lr, 1.0f);
        mat_axpy(nn->layers[l].biases, nn->layers[l].grad_biases, -lr, 1.0f);
        layer_sync_weights_T(&nn->layers[l]);
    }
}

#endif // NN_IMPLEMENTATION

#endif // NN_H_
