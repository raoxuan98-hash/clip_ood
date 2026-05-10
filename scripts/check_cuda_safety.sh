#!/bin/bash
# CUDA 安全检查脚本
# 验证 GPU 环境是否安全可用

echo "=========================================="
echo "CUDA Safety Check"
echo "=========================================="

# 1. 检查 nvidia-smi
echo -n "[1/4] Checking nvidia-smi... "
if nvidia-smi -L &>/dev/null; then
    echo "✅ OK"
else
    echo "❌ FAILED - nvidia-smi error"
    exit 1
fi

# 2. 检查 GPU 3 状态
echo -n "[2/4] Checking GPU 3 status... "
gpu3_status=$(nvidia-smi -L 2>&1 | grep -A1 "GPU 3")
if echo "$gpu3_status" | grep -q "Unknown Error"; then
    echo "⚠️  GPU 3 has error (expected - will be excluded)"
elif echo "$gpu3_status" | grep -q "GPU 3"; then
    echo "⚠️  GPU 3 is online but may be unstable"
else
    echo "✅ GPU 3 not detected"
fi

# 3. 检查可用 GPU
echo -n "[3/4] Checking available GPUs... "
available_gpus=$(nvidia-smi -L 2>/dev/null | grep "GPU" | grep -v "Unknown Error" | wc -l)
if [ "$available_gpus" -ge 5 ]; then
    echo "✅ Found $available_gpus GPUs (>= 5 required)"
else
    echo "❌ Only $available_gpus GPUs available (< 5 required)"
    exit 1
fi

# 4. 测试 CUDA 初始化 (排除 GPU 3)
echo -n "[4/4] Testing CUDA initialization... "
export CUDA_VISIBLE_DEVICES=0,1,2,4,5

python -c "
import torch
try:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count >= 5:
            print(f'✅ OK - CUDA available with {count} devices')
            exit(0)
        else:
            print(f'❌ FAILED - Only {count} CUDA devices')
            exit(1)
    else:
        print('❌ FAILED - CUDA not available')
        exit(1)
except Exception as e:
    print(f'❌ FAILED - {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "❌ CUDA Safety Check FAILED"
    echo "=========================================="
    echo "GPU 3 may be causing issues."
    echo "Try: sudo nvidia-smi --gpu-reset -i 3"
    echo "Or:  sudo reboot"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All Checks Passed!"
echo "=========================================="
echo "Safe to use GPUs: 0, 1, 2, 4, 5"
echo "GPU 3 is excluded as required."
echo "=========================================="
