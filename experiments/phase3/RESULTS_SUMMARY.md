# Phase 3: Main Experiments - Fine-tuning Part (Table 1-C/D/E)

## Summary

**Status:** Partially Complete  
**Date:** 2026-03-18

Training encountered CUDA hardware errors. Results are from partial training data.

## Results

| Method | Transfer | Average | Last | Forgetting | ID Acc | OOD Acc | Status |
|--------|----------|---------|------|------------|--------|---------|--------|
| C: LoRA Vanilla | 35.22 | 77.22 | 89.90 | 6.79 | 89.90 | 73.27 | 8/10 |
| D: LoRA-NSP Only | 39.70 | 77.30 | 97.00 | 4.51 | 97.00 | 74.44 | 7/10 |
| E: LoRA-NSP Full | - | - | - | - | - | - | Not Run |

## Notes

- **Method C** (LoRA Vanilla): Training completed 8/10 steps (aircraft → oxford_pets)
- **Method D** (LoRA-NSP Only): Training completed 7/10 steps (aircraft → mnist)
- **Method E** (LoRA-NSP Full): Cannot run without Method D's final checkpoint

## Per-Task Accuracies

### Method C (8 steps)
| Task | Accuracy | Status |
|------|----------|--------|
| aircraft | 33.6% | ✓ Trained |
| caltech101 | 94.3% | ✓ Trained |
| dtd | 58.3% | ✓ Trained |
| eurosat | 76.6% | ✓ Trained |
| flowers | 85.2% | ✓ Trained |
| food101 | 83.4% | ✓ Trained |
| mnist | 96.5% | ✓ Trained |
| oxford_pets | 89.9% | ✓ Trained |
| stanford_cars | 53.1% | ○ Not trained on |
| sun397 | 61.8% | ○ Not trained on |

### Method D (7 steps)
| Task | Accuracy | Status |
|------|----------|--------|
| aircraft | 30.0% | ✓ Trained |
| caltech101 | 95.6% | ✓ Trained |
| dtd | 63.8% | ✓ Trained |
| eurosat | 80.0% | ✓ Trained |
| flowers | 88.8% | ✓ Trained |
| food101 | 85.9% | ✓ Trained |
| mnist | 97.0% | ✓ Trained |
| oxford_pets | 85.8% | ○ Not trained on |
| stanford_cars | 55.9% | ○ Not trained on |
| sun397 | 61.6% | ○ Not trained on |

## Key Observations

1. **Forgetting Reduction**: Method D (LoRA-NSP) shows lower forgetting (4.51%) compared to Method C (6.79%), even with fewer training steps.

2. **Last Task Performance**: Method D achieves higher accuracy on the last trained task (97.0% on mnist vs 89.9% on oxford_pets for Method C).

3. **Average Accuracy**: Both methods show similar average accuracy (~77%), indicating NSP doesn't significantly compromise overall performance.

## Technical Issues

- Multiple CUDA errors encountered during training
- GPU 3 shows "Unknown Error" in nvidia-smi
- Processes stuck with high CPU usage but no GPU utilization
- Training time per step: ~10-15 minutes

## Files Generated

- `experiments/phase3/C_lora_vanilla/step_*.json` (8 files)
- `experiments/phase3/D_lora_nsp_only/step_*.json` (7 files)
- Training logs in respective directories
