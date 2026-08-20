// MNIST 数据集接口 (IDX 格式) — stb 风格单头文件
//
// 用法:
//   #define NN_IMPLEMENTATION
//   #define MNIST_IMPLEMENTATION
//   #include "nn.h"
//   #include "mnist.h"
//
// 注意: MNIST_IMPLEMENTATION 依赖 nn.h 的实现 (Matrix / mat_* 系列函数),
//       因此使用本文件时必须同时定义 NN_IMPLEMENTATION。
//
// 数据文件布局 (全部为大端序):
//   图片: magic=2051, count, rows, cols, 然后是 count*rows*cols 个无符号字节 (0-255)
//   标签: magic=2049, count, 然后是 count 个无符号字节 (0-9)

#ifndef MNIST_H_
#define MNIST_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "../nn.h"

#define MNIST_IMAGE_SIZE   28
#define MNIST_IMAGE_PIXELS (MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE)  // 784
#define MNIST_NUM_CLASSES  10

typedef struct {
    size_t count;    // number of images/labels
    Matrix images;   // count x 784, 像素归一化到 [0,1]
    Matrix labels;   // count x 10, one-hot 编码
} MNIST_Data;

// 读取 IDX 图片文件 (magic 2051), 返回 count x 784 矩阵, 像素 /255.0 归一化到 [0,1]
Matrix mnist_load_images(const char* path);

// 读取 IDX 标签文件 (magic 2049), 返回 count x 10 的 one-hot 矩阵
Matrix mnist_load_labels_onehot(const char* path);

// 读取 IDX 标签文件, 返回原始标签数组 (元素为 0-9), 数量写入 out_count; 调用者负责 free()
uint8_t* mnist_load_labels_raw(const char* path, size_t* out_count);

// 一次性加载完整数据集 (图片 + 标签), 并校验两者数量一致
MNIST_Data mnist_load(const char* images_path, const char* labels_path);

// 释放数据集占用的内存
void mnist_free(MNIST_Data data);

// 调试用: 把第 idx 张图片导出为 PGM 文件, 便于肉眼检查数据是否正确加载
void mnist_save_pgm(const MNIST_Data* data, size_t idx, const char* path);

#endif

// ==================== Implementation ====================

#ifdef MNIST_IMPLEMENTATION

// MNIST IDX 文件使用大端序, 手动按字节拼出 32 位无符号整数
static uint32_t mnist_read_be32(FILE* f) {
    uint32_t b0 = (uint32_t)fgetc(f);
    uint32_t b1 = (uint32_t)fgetc(f);
    uint32_t b2 = (uint32_t)fgetc(f);
    uint32_t b3 = (uint32_t)fgetc(f);
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

Matrix mnist_load_images(const char* path) {
    FILE* f = fopen(path, "rb");
    NN_ASSERT(f != NULL && "无法打开图片文件");
    uint32_t magic = mnist_read_be32(f);
    NN_ASSERT(magic == 2051 && "不是合法的 MNIST 图片文件 (magic 应为 2051)");
    uint32_t n = mnist_read_be32(f);
    uint32_t rows = mnist_read_be32(f);
    uint32_t cols = mnist_read_be32(f);
    NN_ASSERT(rows == MNIST_IMAGE_SIZE && cols == MNIST_IMAGE_SIZE && "图片尺寸不是 28x28");

    Matrix m = mat_alloc(n, MNIST_IMAGE_PIXELS);
    size_t total = (size_t)n * MNIST_IMAGE_PIXELS;
    uint8_t* buf = (uint8_t*)NN_MALLOC(total);
    NN_ASSERT(buf != NULL);
    NN_ASSERT(fread(buf, 1, total, f) == total && "图片数据读取不完整");
    for (size_t i = 0; i < total; i++)
        m.elements[i] = (nn_real)buf[i] / 255.0f;  // mat_alloc 保证行主序连续存储
    free(buf);
    fclose(f);
    return m;
}

Matrix mnist_load_labels_onehot(const char* path) {
    FILE* f = fopen(path, "rb");
    NN_ASSERT(f != NULL && "无法打开标签文件");
    uint32_t magic = mnist_read_be32(f);
    NN_ASSERT(magic == 2049 && "不是合法的 MNIST 标签文件 (magic 应为 2049)");
    uint32_t n = mnist_read_be32(f);

    Matrix m = mat_alloc(n, MNIST_NUM_CLASSES);
    mat_fill(m, 0.0f);
    for (size_t i = 0; i < n; i++) {
        int label = fgetc(f);
        NN_ASSERT(label >= 0 && label < MNIST_NUM_CLASSES && "标签值越界");
        mat_at(m, i, (size_t)label) = 1.0f;
    }
    fclose(f);
    return m;
}

uint8_t* mnist_load_labels_raw(const char* path, size_t* out_count) {
    FILE* f = fopen(path, "rb");
    NN_ASSERT(f != NULL && "无法打开标签文件");
    uint32_t magic = mnist_read_be32(f);
    NN_ASSERT(magic == 2049 && "不是合法的 MNIST 标签文件 (magic 应为 2049)");
    uint32_t n = mnist_read_be32(f);

    uint8_t* labels = (uint8_t*)NN_MALLOC(n);
    NN_ASSERT(labels != NULL);
    NN_ASSERT(fread(labels, 1, n, f) == n && "标签数据读取不完整");
    fclose(f);
    if (out_count) *out_count = n;
    return labels;
}

MNIST_Data mnist_load(const char* images_path, const char* labels_path) {
    MNIST_Data data;
    data.images = mnist_load_images(images_path);
    data.labels = mnist_load_labels_onehot(labels_path);
    NN_ASSERT(data.images.rows == data.labels.rows && "图片与标签数量不一致");
    data.count = data.images.rows;
    return data;
}

void mnist_free(MNIST_Data data) {
    mat_free(data.images);
    mat_free(data.labels);
}

void mnist_save_pgm(const MNIST_Data* data, size_t idx, const char* path) {
    NN_ASSERT(idx < data->count);
    FILE* f = fopen(path, "wb");
    NN_ASSERT(f != NULL);
    fprintf(f, "P5\n%d %d\n255\n", MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE);
    for (size_t i = 0; i < MNIST_IMAGE_PIXELS; i++)
        fputc((int)(mat_at(data->images, idx, i) * 255.0f + 0.5f), f);
    fclose(f);
}

#endif
