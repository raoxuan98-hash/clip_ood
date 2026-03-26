#!/bin/bash
# Test CUDA availability

export CUDA_VISIBLE_DEVICES=0,1,2,4,5

python -c "
import torch
print('Testing CUDA...')
try:
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Device count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('CUDA not available')
except Exception as e:
    print(f'Error: {e}')
"
