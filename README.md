# CIFAR-10 ResNet under a 5M-Parameter Budget

Deep Learning Project 1, Fall 2026, NYU Tandon. Kaggle competition: highest test accuracy on a held-out CIFAR-10 set using a ResNet with **at most 5,000,000 trainable parameters**, trained from scratch.

**Team:** _Name 1, Name 2, Name 3_

## Results

| Item | Value |
|---|---|
| Trainable parameters | **4,815,940** (limit 5,000,000) |
| Validation accuracy (EMA + flip TTA, 5k held-out train images) | _XX.XX%_ |
| Kaggle test accuracy / leaderboard rank | _XX.XX% / rank N_ |
| Training time | _N hours on 1x GPU model_ |

The executed notebook (`CIFAR10_ResNet_Final_HPC.ipynb`) prints the final parameter count and accuracy, and shows the full training log and curves.

## Model

CIFAR-style ResNet: 3x3 conv stem, 4 residual stages with 2 basic blocks each, global average pool, dropout 0.1, linear head.

| Variable | Value |
|---|---|
| N (residual layers) | 4 |
| B_i (blocks per layer) | 2, 2, 2, 2 |
| C_i (channels) | 42, 84, 168, 336 |
| F_i (conv kernel) | 3x3 |
| K_i (skip kernel) | 1x1 projection conv + BN when shape changes, identity otherwise |
| P (average pool) | global, over the 4x4 final map |

Residual block: `ReLU(S(x) + F(x))`, where `F = conv-BN-ReLU-conv-BN`. Stages 2 to 4 downsample with stride 2 in their first block.

## Training recipe

| Component | Setting |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 (not on BN/bias) |
| LR schedule | OneCycle: 15% warm-up to 0.1, cosine decay |
| Loss | Cross-entropy, label smoothing 0.1 |
| Augmentation | RandomCrop (reflect pad 4), horizontal flip, TrivialAugment, CutMix (p=0.5, off for last 10 epochs) |
| Weight averaging | EMA, decay 0.999; the final model is the EMA copy at the last epoch |
| Precision / batch / epochs | AMP, batch 128, 200 epochs |
| Inference | Horizontal-flip test-time augmentation |
| Data split | 45,000 train / 5,000 validation, from the provided training images only |

No pretrained weights and no external data are used. The Kaggle test images are used only to generate the final predictions.

## Repository layout

```
.
├── CIFAR10_ResNet_Final_HPC.ipynb   # full pipeline, run top to bottom
├── README.md
├── requirements.txt
├── curves.png                       # training curves from the final run
└── submission.csv                   # final Kaggle predictions (optional)
```

Data (`data.zip`) and checkpoints (`ckpt.pt`, `final_ema.pt`) are not committed.

## Reproducing the results

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Put the competition `data.zip` in the working folder. The notebook expects `/scratch/sm14460/data.zip`; change `WORK` in the first code cell for another location. The zip contains `train_images/`, `test_images/`, `train_labels.csv`, `test.csv` and `sample_submission.csv`.
3. Open the notebook on a GPU machine and run all cells top to bottom (`jupyter nbconvert --to notebook --execute CIFAR10_ResNet_Final_HPC.ipynb` also works).
4. Outputs are written to the working folder: `submission.csv`, `curves.png`, `final_ema.pt`.

Notes:
- Seeds are fixed (`seed=0`), but GPU kernels are not bit-exact, so small accuracy differences between runs are expected.
- Training saves `ckpt.pt` every epoch and resumes automatically after an interruption. Delete `ckpt.pt` for a fresh run.
- A full run takes about _N_ hours on _GPU model_.

## requirements.txt

```
torch>=2.1
torchvision>=0.16
numpy
pandas
pillow
matplotlib
torchsummary   # optional, prints the layer table
```

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- S. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
- S. Muller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021.
- L. Smith, N. Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.* 2018.
- C. Szegedy et al. *Rethinking the Inception Architecture for Computer Vision* (label smoothing). CVPR 2016.
- PyTorch `torch.optim.swa_utils` for weight EMA.
