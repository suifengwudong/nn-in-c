$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path "build" -Force | Out-Null

# tests
# gcc -g -Wall -Wextra -o build/nn_tests.exe tests/test_backprop.c
# Write-Host ""
# Write-Host "Running nn_tests.exe..."
# & build/nn_tests.exe

# examples

# XOR 训练
gcc -O2 -Wall -Wextra -o build/xor_nn.exe example/xor_nn.c
Write-Host ""
Write-Host "Running xor_nn.exe..."
& build/xor_nn.exe

# MNIST 训练 (优化构建, 速度较快; 用法: mnist_nn [epochs] [batch_size] [learning_rate])
# gcc -O2 -Wall -Wextra -o build/mnist_nn.exe example/mnist_nn.c
# gcc -O2 -Wall -Wextra -DNN_BATCH -o build/mnist_nn.exe example/mnist_nn.c
# Write-Host ""
# Write-Host "Running mnist_nn.exe (default: 5 epochs)..."
# & build/mnist_nn.exe

gcc -O3 -march=native -Wall -Wextra -DNN_BATCH -o build/mnist_batch_nn.exe example/mnist_nn.c
Write-Host ""
Write-Host "Running mnist_batch_nn.exe (default: 5 epochs)..."
& build/mnist_batch_nn.exe

# python .\draw_plot.py