$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path "build" -Force | Out-Null

# tests
# gcc -g -Wall -Wextra -o build/nn_tests.exe tests/test_backprop.c
# Write-Host ""
# Write-Host "Running nn_tests.exe..."
# & build/nn_tests.exe

# MNIST 训练 (优化构建, 速度较快; 用法: mnist_nn [epochs] [batch_size] [learning_rate])
gcc -O2 -Wall -Wextra -o build/mnist_nn.exe minst_nn.c
Write-Host ""
Write-Host "Running mnist_nn.exe (default: 5 epochs)..."
& build/mnist_nn.exe
