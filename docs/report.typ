#import "@preview/easy-paper:0.2.1": *

#show: project.with(
  title: [深度学习C实现笔记],
  author: "HCX",
  date: auto
)

= 推导

== 反向传播

根据神经网络，可以记 $W_(i)$，$bold(b)_(i)$ 为第 $i$ 层到第 $i+1$ 层的神经元权重，记中间结果激活值为 $bold(a)_(i)$，则对每一层都有
$ bold(a)_(i+1) = sigma(W_(i)bold(a)_(i) + bold(b)_(i)) eq.delta sigma(bold(t)_(i)) $
其中，$sigma(dot)$ 为激活函数，$bold(t)_(i)$ 为 weighted-sum 值。根据损失函数 $ell(hat(bold(y)), bold(y)) = 1/n (hat(bold(y)) - bold(y))^T (hat(bold(y)) - bold(y))$，则由 $hat(bold(y)) = bold(a)_(n+1)$ 得
$ (partial ell) / (partial bold(a)_(n+1)) = (partial ell) / (partial hat(bold(y))) = 2/n (hat(bold(y)) - bold(y)) = 2/n (bold(a)_(n+1) - bold(y)) $
$ (partial bold(a)_(i+1)) / (partial bold(a)_(i)) = (partial bold(a)_(i+1)) / (partial bold(t)_(i)) (partial bold(t)_(i)) / (partial bold(a)_(i)) = sigma'(bold(t)_(i)) dot.o W_i $
$ (partial bold(t)_(i)) / (partial W_(i)) = bold(a)_(i)^T, quad (partial bold(t)_(i)) / (partial bold(b)_(i)) = I $
则定义误差项 $bold(delta)_(i) = (partial ell) / (partial bold(t)_(i))$。在输出层 $i = n$ 处，由 $hat(bold(y)) = bold(a)_(n+1) = sigma(bold(t)_(n))$ 得
$ bold(delta)_(n) = (partial ell) / (partial bold(a)_(n+1)) dot.o sigma'(bold(t)_(n)) = 2/n (bold(a)_(n+1) - bold(y)) dot.o sigma'(bold(t)_(n)) $
其中 $dot.o$ 表示逐元素（Hadamard）乘积。递推得
$ bold(delta)_(i) = sigma'(bold(t)_(i)) dot.o (W_(i+1)^T bold(delta)_(i+1)), quad i = n-1, ..., 1 $
结合 $(partial bold(t)_(i)) / (partial W_(i)) = bold(a)_(i)^T$ 与 $(partial bold(t)_(i)) / (partial bold(b)_(i)) = I$，即得各层参数的梯度
$ (partial ell) / (partial W_(i)) = bold(delta)_(i) bold(a)_(i)^T, quad (partial ell) / (partial bold(b)_(i)) = bold(delta)_(i) $


= 编程经验

== C 语言相关

=== X-macro

```c

// ==================== Activation System ====================

static inline nn_real sigmoidf(nn_real x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline nn_real reluf(nn_real x) {
    return x > 0.0f ? x : 0.0f;
}

static inline nn_real geluf(nn_real x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

#define ACTIVATIONS \
    X(ACT_SIGMOID, sigmoidf) \
    X(ACT_RELU,    reluf)    \
    X(ACT_GELU,    geluf)

#define X(id, name) id,
typedef enum { ACTIVATIONS } ActivationType;
#undef X

static inline nn_real activate(ActivationType type, nn_real x) {
    switch (type) {
        #define X(id, name) case id: return name(x);
        ACTIVATIONS
        #undef X
    }
    return x;
}
```

X-macro 技术可以用来定义一组相关的宏或函数，避免重复代码，提高可维护性。

=== 视图式的矩阵内存管理

```c
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    nn_real *elements;
} Matrix;
```

使用 `stride` 来表示每行的跨度，这样可以方便地创建子矩阵或行视图，而不需要复制数据。