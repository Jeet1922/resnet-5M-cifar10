# Architecture and Implementation Design

This document describes the model, the training pipeline and the design decisions behind the CIFAR-10 ResNet (4,815,940 trainable parameters, limit 5,000,000).

## 1. Model overview

A CIFAR-style ResNet: a 3x3 conv stem, four residual stages with two basic blocks each, global average pooling and a linear classifier. It outputs raw logits; softmax is applied only inside the loss.

```
Input 3x32x32
  |
Stem: Conv3x3(3->42) -> BN -> ReLU                      42x32x32
  |
Stage 1: 2 x BasicBlock(42),  stride 1                  42x32x32
Stage 2: 2 x BasicBlock(84),  first block stride 2      84x16x16
Stage 3: 2 x BasicBlock(168), first block stride 2     168x8x8
Stage 4: 2 x BasicBlock(336), first block stride 2     336x4x4
  |
Global average pool                                     336
Dropout(0.1)
Linear(336 -> 10)                                       10 logits
```

### Basic residual block

```
x ---------------------------------------+
|                                         |  S(x): identity, or
Conv3x3 (stride s) -> BN -> ReLU          |        Conv1x1 (stride s) -> BN
Conv3x3 (stride 1) -> BN   = F(x)         |        when stride or channels change
|                                         |
+---------------- add <-------------------+
                   |
                 ReLU        out = ReLU(S(x) + F(x))
```

Both branches must have the same shape before the addition. Convolutions have no bias because BatchNorm follows each one.

### Mapping to the handout's design variables

| Variable | Value |
|---|---|
| N, residual layers | 4 |
| B_i, blocks per layer | 2, 2, 2, 2 |
| C_i, channels | 42, 84, 168, 336 |
| F_i, conv kernel | 3x3 |
| K_i, skip kernel | 1x1 projection (identity where shapes match) |
| P, pool | global average over the 4x4 map |

## 2. Parameter budget

| Component | Trainable parameters |
|---|---|
| Stem (conv + BN) | 1,218 |
| Stage 1 (2 blocks) | 63,840 |
| Stage 2 (2 blocks + projection) | 226,632 |
| Stage 3 (2 blocks + projection) | 904,848 |
| Stage 4 (2 blocks + projection) | 3,616,032 |
| Classifier (336 -> 10) | 3,370 |
| **Total** | **4,815,940** |

Convolution cost grows with k² · C_in · C_out, so width dominates the budget: stage 4 alone holds about 75% of all parameters. The widths 42/84/168/336 were chosen to use most of the 5M budget (about 184k spare) while keeping the standard 4-stage ResNet-18 layout. BatchNorm running means and variances are buffers, not trainable parameters, so they don't count toward the limit. The notebook asserts `params <= 5_000_000` before training starts.

## 3. Data pipeline

```
data.zip -> train_images/*.png + train_labels.csv, test_images/*.png, test.csv, sample_submission.csv
   |
   +-- all PNGs decoded once into uint8 NumPy arrays (kept in RAM)
   +-- train images: 45k train / 5k validation split (fixed seed)
   +-- test images loaded in sample_submission.csv order
```

| Split | Transform |
|---|---|
| Train | RandomCrop(32, pad 4, reflect), horizontal flip, TrivialAugment, normalize |
| Validation / test | Normalize only (plus flip TTA at inference) |

Normalization uses the standard CIFAR-10 channel means and standard deviations.

## 4. Training pipeline

| Component | Choice | Purpose |
|---|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov | Reliable convergence for BN convnets |
| Weight decay | 5e-4 on weights only | Regularization without disturbing BN/bias |
| LR schedule | OneCycle: 15% warm-up to 0.1, cosine decay | Fast, stable training |
| Loss | Cross-entropy, label smoothing 0.1 | Less overconfidence |
| CutMix | p = 0.5, switched off for the last 10 epochs | Strong regularizer; clean fine-tuning at the end |
| EMA | decay 0.999 over weights and BN buffers | Smoother final weights; the final model is the EMA copy |
| Mixed precision | AMP + channels_last | Faster GPU training |
| Checkpointing | `ckpt.pt` every epoch, automatic resume | Survives HPC walltime limits |

Each step: load batch, optionally apply CutMix, forward pass under AMP, loss (mixed-label loss when CutMix is used), backward with gradient scaling, SGD step, scheduler step, EMA update.

## 5. Inference and submission

1. Take the EMA model from the last epoch (no checkpoint selection on validation data).
2. Predict logits for the original and horizontally flipped image, sum them, take the argmax (flip TTA).
3. Copy `sample_submission.csv`, fill the label column with the predictions and save `submission.csv`, which preserves the required IDs and order.

## 6. Design decisions

- **Basic blocks, not bottlenecks.** At these widths they give the best accuracy per parameter and are simple to verify.
- **No max-pool in the stem.** CIFAR images are only 32x32, so keeping full resolution in the first stage preserves detail.
- **Projection shortcuts only where needed.** Identity skips cost nothing; 1x1 convolutions appear only at the three downsampling points.
- **Regularization over size.** With a 5M cap, accuracy gains come mostly from augmentation, label smoothing and EMA rather than extra capacity.
- **Validation split instead of the Kaggle test set.** All decisions use the 5k validation images; test images are only used for the final predictions.

## 7. Compliance with the project rules

| Rule | How it is met |
|---|---|
| At most 5M trainable parameters | 4,815,940, enforced by an assert |
| ResNet with skip connections | Residual blocks with identity or 1x1 projection skips |
| Train from scratch | Random initialization; no pretrained weights |
| No outside or larger datasets | Only the provided competition images |
| No tuning on the test set | Model selection uses the validation split only |

## 8. Modifying the model

Change `CFG` in the first code cell:

- `blocks=(2, 2, 2, 2)`: blocks per stage
- `channels=(42, 84, 168, 336)`: width per stage
- `dropout`, `epochs`, `lr`, `wd`, `smooth`, `cutmix_prob`, `ema_decay`: training settings

After any change, check the printed parameter count before training. Delete `ckpt.pt` before a fresh run with new settings.

## 9. Files

| File | Role |
|---|---|
| `CIFAR10_ResNet_Final_HPC.ipynb` | Full pipeline: data loading, model, training, results, submission |
| `README.md` | Results and reproduction instructions |
| `ARCHITECTURE.md` | This document |
| `requirements.txt` | Python dependencies |
