# Comparative Deep Learning Framework for Localized Road Damage Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: RDD2022](https://img.shields.io/badge/dataset-RDD2022-green.svg)](https://doi.org/10.1002/gdj3.260)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

A controlled, single-protocol comparison of three deep learning architecture
families for **patch-level road damage classification** on the RDD2022 dataset,
evaluated jointly on classification performance, computational efficiency,
class-level error structure and visual interpretability.

**The contribution is the comparison, not a new architecture.** Published
road-damage work is dominated by architectural proposals evaluated under each
author's own pipeline, which makes cross-paper comparison between architectures
confounded by differences in preprocessing, resolution, augmentation, splitting
and evaluation. This project holds every non-architectural factor constant and
varies only the backbone.

---

## Headline results

Held-out test set: **7,856 patches** from a source-image-level split that no
model saw during training or selection.

| Model | Accuracy | **Macro-F1** | Weighted-F1 | Params | Size | Latency (bs=1) | Train time |
|---|---|---|---|---|---|---|---|
| Custom CNN | 0.8769 | 0.8593 | 0.8783 | 1.17 M | 4.7 MB | **0.84 ms** | 190.8 min |
| **ResNet50** | **0.9413** | **0.9347** | **0.9412** | 23.52 M | 94.4 MB | 6.45 ms | 215.8 min |
| EfficientNet-B0 | 0.9381 | 0.9321 | 0.9382 | **4.01 M** | **16.3 MB** | 6.70 ms | **141.9 min** |

Macro-F1 is the primary metric, not accuracy — it weights every class equally, so
failure on the minority pothole class cannot be masked by the majority crack
classes.

### Per-class F1

| Model | D00 Longitudinal | D10 Transverse | D20 Alligator | D40 Pothole |
|---|---|---|---|---|
| Custom CNN | 0.8999 | 0.9269 | 0.8299 | 0.7803 |
| ResNet50 | **0.9531** | **0.9583** | **0.9040** | **0.9233** |
| EfficientNet-B0 | 0.9495 | 0.9560 | 0.9011 | 0.9217 |

---

## Findings

### 1. EfficientNet-B0 is the practical recommendation

It reaches within **0.26 points** of ResNet50's macro-F1 using **5.9× fewer
parameters** and **5.8× less disk**. For deployment on smartphones, dashcams or
roadside units — the contexts that motivate this work — that is not a close
second, it is the better engineering choice.

### 2. Parameter count is not latency

EfficientNet-B0 is *slower* at inference than ResNet50 (6.70 ms vs 6.45 ms)
despite having a sixth of the parameters. Depthwise separable convolutions reduce
FLOPs and parameters but are memory-bandwidth-bound and map poorly onto tensor
cores. **The efficiency gain here is in memory and storage, not speed.** This is
an easy conclusion to get wrong by reporting parameter count alone, which is why
§17.2 of the protocol requires latency to be measured directly.

### 3. The value of pretraining scales with class difficulty

The gap between the from-scratch CNN and ResNet50 is not uniform:

| Class | Gap (macro-F1 points) |
|---|---|
| D10 Transverse crack | +3.1 |
| D00 Longitudinal crack | +5.3 |
| D20 Alligator crack | +7.4 |
| **D40 Pothole** | **+14.3** |

Transverse cracks are a strong oriented line — a feature any CNN learns from
scratch. Potholes require richer texture representation, and D40 is also the
scarcest class (6,544 instances), so ImageNet features are compensating for data
scarcity as well as complexity.

### 4. One error mode dominates, and it is shared

Every model's largest confusion is **D20 → D00** (10% / 8% / 7%). Alligator
cracking develops *from* interconnected longitudinal cracks, so early-stage D20
genuinely contains D00 morphology. Part of this error is likely label ambiguity
rather than model failure — the confusion structure is consistent across
architectures, which suggests it is intrinsic to the task.

---

## Methodology highlights

### Leakage-resistant splitting

RDD2022 is a detection dataset. Converting it to classification means cropping
each annotated box into its own image — and **61.7% of annotated images contain
more than one damage instance** (up to 44). Pooling the patches and splitting
randomly would scatter siblings from the same road scene, the same pavement, the
same lighting and the same camera across train and test. A model could then score
highly by recognising scenes rather than damage.

This repository splits at the **source-image level, before patches exist**, and
goes further: consecutive frames within a country are bucketed in groups of five,
because road imagery is captured as a sequence from a moving vehicle and adjacent
filenames often show the same physical crack a metre further along.

Both pipeline stages assert zero leakage and must print `PASS`.

Reported scores are therefore **lower** than naive patch-level splitting would
produce. That is the intended consequence.

### Filter thresholds chosen by measurement, not convention

Quality filters were selected by measuring **per-class retention**, after the
obvious defaults turned out to be class-selective:

| Setting | D00 | D10 | D20 | D40 | Verdict |
|---|---|---|---|---|---|
| `min_side ≥ 24`, `aspect ≤ 8` | 93% | **69%** | 100% | 82% | Rejected |
| `min_side ≥ 16`, `aspect ≤ 20` | 98% | 91% | 100% | 96% | **Adopted** |

Transverse cracks are inherently elongated — often 16 px tall and 400 px wide. A
strict aspect limit deletes a third of D10 and almost nothing else, converting a
quality filter into a silent, class-biased deletion that would then be
misattributed to the model.

### Class set justified arithmetically

RDD2022's annotation files contain more than the four documented damage types.
The extras are road markings (D43 crosswalk blur, D44 white line blur), fixtures
(D50 manhole covers), repairs, and construction joints — none of which are
damage. Restricting to D00/D10/D20/D40 yields **55,006 instances**, reconciling
exactly with the "more than 55,000" figure published in the RDD2022 data article.

Note that the non-canonical codes sit almost entirely in Japan and India, so
dropping them is **not** a geographically neutral operation.

### Augmentation that respects the label

Horizontal flip, ±7° affine jitter and mild colour jitter. Vertical flip and
large-angle rotation are **excluded by design**: crack orientation is
class-defining, so a longitudinal crack rotated 90° is visually indistinguishable
from a transverse crack. That augmentation would corrupt the label itself.

### What is held constant

| Factor | Status |
|---|---|
| Architecture, pretraining regime | **Varied** — the object of study |
| Patch extraction, filtering, resolution | Fixed |
| Augmentation policy | Fixed |
| Train/val/test split (seed 42) | Fixed, persisted, reused verbatim |
| Class-balancing (weighted loss) | Fixed |
| Optimiser (AdamW), cosine schedule, batch size, epoch budget | Fixed |
| Model selection criterion (validation macro-F1) | Fixed |
| Evaluation hardware | Fixed |
| Normalisation statistics | **Documented deviation** — see below |

The one deliberate deviation: pretrained backbones use ImageNet normalisation
statistics, the from-scratch CNN uses dataset statistics. Feeding a pretrained
backbone the wrong input distribution would handicap it for reasons unrelated to
architecture.

---

## Dataset

[**RDD2022**](https://doi.org/10.1002/gdj3.260) (Arya et al., 2024) — 47,420 road
images from six countries across seven capture subsets, with over 55,000
annotated damage instances. Download from
[figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)
(~13 GB, CC BY-SA 4.0).

Only the `train` folders are used — annotations for the released `test` folders
were withheld for the CRDDC'2022 leaderboard and have never been published.

### Pipeline statistics

```
38,385 annotation files across 7 subsets
   ↓  parse, integrity-check
65,711 instances from 26,661 annotated images
   (11,724 images carry no annotation — expected, ~31%)
   (1 rejected for non-positive area; 0 malformed XML, 0 missing images)
   ↓  restrict to the four canonical damage types
55,006 instances
   ↓  quality filtering (min_side ≥ 16 px, aspect ≤ 20:1)
53,242 patches at 224×224   (1,558 + 206 rejected)
   ↓  source-image-level split, seed 42, group size 5
train 37,468  |  val 7,918  |  test 7,856
```

### Class and country distribution

| Subset | D00 | D10 | D20 | D40 | Total |
|---|---|---|---|---|---|
| Japan | 4,049 | 3,979 | 6,198 | 2,243 | 16,469 |
| Norway | 8,570 | 1,730 | 468 | 461 | 11,229 |
| United States | 6,750 | 3,295 | 834 | 135 | 11,014 |
| India | 1,555 | 68 | 2,021 | 3,187 | 6,831 |
| China (motorbike) | 2,678 | 1,096 | 641 | 235 | 4,650 |
| China (drone) | 1,426 | 1,263 | 293 | 86 | 3,068 |
| Czech | 988 | 399 | 161 | 197 | 1,745 |
| **Total** | **26,016** | **11,830** | **10,616** | **6,544** | **55,006** |

Overall class imbalance is a mild 4:1. The **geographic** skew is far more severe
and more interesting: India has 68 transverse cracks against 2,021 alligator
cracks; the US has 135 potholes; Japan holds 58% of all alligator cracking. These
are not merely different countries — they are different damage profiles, so a
model trained on the pooled set learns a distribution no individual country
exhibits.

---

## Repository structure

```
.
├── 1_custom_cnn.ipynb          # Model 1 — from-scratch baseline
├── 2_resnet50.ipynb            # Model 2 — ImageNet transfer learning
├── 3_efficientnet_b0.ipynb     # Model 3 — ImageNet transfer learning
├── Dataset/                    # unpacked RDD2022 (not tracked)
├── data/                       # generated: index, splits, patches, manifest
├── reports/                    # generated: figures, metrics, fingerprints
└── runs/                       # generated: checkpoints per model
```

Sections 1–7 of the three notebooks are **byte-for-byte identical** — the
preprocessing is verified by hash, not by assertion. Each notebook emits a
**preprocessing fingerprint** (an order-independent MD5 of the patch manifest
plus the protocol constants). All three must match before any result is
comparable; a mismatch means preprocessing diverged and the comparison is invalid.

---

## Reproducing

### Requirements

```bash
conda create -n rdd python=3.11 -y
conda activate rdd

# Get the exact command from https://pytorch.org/get-started/locally
# Do NOT install the CUDA Toolkit separately — the wheels bundle their own runtime.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install "numpy<2.3" pandas opencv-python pillow scikit-learn \
            matplotlib seaborn tqdm pyyaml grad-cam ipykernel ipywidgets
```

Verify before proceeding:

```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Running

1. Unpack RDD2022 into `Dataset/` beside the notebooks.
2. Open any notebook and run Sections 1–7 (~45 min, mostly patch extraction).
   Both leakage checks must print `PASS`.
3. Compare the Section 7 fingerprint against your collaborators' before training.
4. Run Sections 8–11 to train, evaluate and interpret that model.
5. Run Section 12 once all three `reports/result_*.csv` exist to produce the
   comparison table, accuracy-versus-cost plot and combined confusion matrices.

Parsing, splitting, extraction and training all cache — re-running a completed
notebook reproduces results in seconds rather than hours.

### Hardware used

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6.44 GB, compute capability 8.9 |
| Framework | PyTorch 2.6.0+cu124, CUDA 12.4 |
| Precision | Mixed (FP16 autocast + GradScaler) |
| Batch size | 32 (peak VRAM 1.41 / 2.97 / ~3 GB) |
| DataLoader workers | 0 |

All three models were trained on the same machine with the same worker count, so
training-time and latency figures **are** directly comparable. Peak VRAM stayed
under 3 GB, so batch size was constrained by the protocol's smallest common
denominator rather than by memory.

### Crash safety

Training writes `last.pt` every epoch with full model, optimiser, scheduler and
AMP scaler state, alongside a flushed `history.csv`. An interruption costs one
epoch; re-running resumes automatically. This was added after a mid-training
power loss destroyed a 30-epoch run.

---

## Limitations

- **Task scope.** Classifies damage type within an already-localized region. It
  does not detect damage in full scenes, segment it at pixel level, or estimate
  severity, extent or repair urgency.
- **92.8% of patches are upscaled.** Most damage regions are smaller than
  224×224, so the majority of training pixels are interpolated rather than
  captured. This also creates a mild domain shift for the ImageNet-pretrained
  models, whose weights were learned on sharp native-resolution photographs.
- **Annotation-bounded.** Localization is supplied by the official bounding
  boxes; classification quality is capped by their consistency. One typo
  (`D0w0` → `D00`) and one non-positive-area box were found and logged.
- **Thin country-class cells.** India contributes only 68 D10 instances and
  China_Drone 86 D40. Per-country per-class metrics for these are reported with
  sample sizes and should not be read as reliable estimates. More fundamentally,
  the pooled model's representation of these combinations is built almost
  entirely from other countries — a property of RDD2022, not of the design.
- **Single run per configuration.** Compute budget did not allow repeated runs,
  so run-to-run variance is unquantified.
- **ResNet50 overfits.** Train macro-F1 0.998 against validation 0.940.
  Validation loss decreased monotonically rather than diverging, and test
  performance (0.9347) tracked validation (0.9420) closely, so the gap was benign
  — but stronger augmentation or weight decay may yield further gains.
- **Grad-CAM is coarse.** It reflects the final convolutional layer and provides
  evidence about attention, not a causal explanation.
- **No deployment claim.** Latency is measured on development hardware. On-device
  benchmarking is future work.

---

## Future work

Severity estimation; full-image detection and segmentation; cross-country domain
adaptation; weather and illumination robustness; on-device benchmarking;
quantization, pruning and distillation; multimodal fusion with LiDAR; GIS
integration for maintenance prioritisation; extension of the same controlled
protocol to vision transformers and state-space architectures.

---

## Citation

If you use this code, please cite the dataset:

```bibtex
@article{arya2024rdd2022,
  title   = {RDD2022: A multi-national image dataset for automatic road damage detection},
  author  = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and
             Toshniwal, Durga and Sekimoto, Yoshihide},
  journal = {Geoscience Data Journal},
  volume  = {11}, number = {4}, pages = {846--862}, year = {2024},
  doi     = {10.1002/gdj3.260}
}
```

Key references for the architectures compared:

- He et al. (2016), *Deep residual learning for image recognition*, CVPR. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
- Tan & Le (2019), *EfficientNet: Rethinking model scaling for CNNs*, ICML.
- Selvaraju et al. (2020), *Grad-CAM*, IJCV. [doi:10.1007/s11263-019-01228-7](https://doi.org/10.1007/s11263-019-01228-7)
- Pardeshi et al. (2025), *Patch-based self-supervised learning for road damage classification*, ISPRS Annals. [doi:10.5194/isprs-annals-X-5-W2-2025-459-2025](https://doi.org/10.5194/isprs-annals-X-5-W2-2025-459-2025)

---

## Authors

| Name | Registration |
|---|---|
| Aaryan Paranjape | 230911168 |
| Aryan Chandra | 230911174 |
| Divyanshu Jain | 230911198 |

---

## License

Code released under the MIT License. The RDD2022 dataset is distributed
separately by its authors under CC BY-SA 4.0 and is not redistributed here.
