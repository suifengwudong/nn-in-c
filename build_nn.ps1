$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path "build" -Force | Out-Null

# XOR 示例 (调试构建)
# gcc -g -Wall -Wextra -o build/nn_xor.exe nn_xor.c
# Write-Host "Running the program..."
# & build/nn_xor.exe

# MNIST 训练 (优化构建, 速度较快; 用法: mnist_nn [epochs] [batch_size] [learning_rate])
gcc -O2 -Wall -Wextra -o build/mnist_nn.exe minst_nn.c
Write-Host ""
Write-Host "Running mnist_nn.exe (default: 5 epochs)..."
& build/mnist_nn.exe
