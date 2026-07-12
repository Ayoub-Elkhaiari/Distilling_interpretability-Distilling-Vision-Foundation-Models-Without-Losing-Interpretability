# Distilling-Interpretability: Does Model Compression Preserve Explanations?

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-ViT--Small-orange)](https://github.com/huggingface/pytorch-image-models)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2)](https://github.com/shap/shap)
[![Colab](https://img.shields.io/badge/Google%20Colab-Compatible-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Oxford--IIIT%20Pet-4169E1)](https://www.robots.ox.ac.uk/~vgg/data/pets/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-First%20Iteration%20%E2%80%94%20WIP-yellow)]()

**Research question:** does knowledge distillation preserve the interpretability of a vision foundation model's decisions, not just its accuracy?

This project distills a frozen ViT-Small teacher into two compact student architectures on the Oxford-IIIT Pet dataset, then goes beyond standard accuracy/efficiency reporting to ask whether the student's explanations (Grad-CAM, SHAP) stay aligned with the teacher's own reasoning (attention rollout). It is built as an executable research notebook — sections, tables, and figures are structured to export directly into a short workshop-style manuscript.

> This is a first, single-seed, Colab-scale iteration. The pipeline is designed to scale to multi-seed, multi-architecture studies with additional compute — see [Limitations](#-limitations--future-work).

---

## ✨ Features

- Frozen ViT-Small teacher (ImageNet-pretrained) adapted with a linear probe on Oxford-IIIT Pet (37 classes)
- Knowledge distillation into two student architectures: a compact custom CNN and MobileNetV2
- Full efficiency suite: accuracy, calibration (ECE), FLOPs, latency, throughput, disk size, compression/speedup ratios
- Explainability comparison across three methods: attention rollout (teacher), Grad-CAM (student), SHAP (student)
- Quantitative explanation agreement via top-20% region IoU and cosine similarity
- Deletion-based faithfulness testing for both teacher and students
- KD hyperparameter ablation (temperature × alpha)
- Reproducibility: logged seeds, package versions, hardware info, Google Drive checkpoint/data caching
- Publication-ready outputs: PNG/PDF figures at 300 DPI, LaTeX-exportable result tables, raw CSV metrics

---

## 🧱 Tech Stack

### Modeling
- PyTorch
- timm (`vit_small_patch16_224`)
- torchvision (MobileNetV2, Oxford-IIIT Pet loader)

### Explainability
- Custom attention rollout (ViT)
- Custom Grad-CAM (CNN students)
- SHAP (`GradientExplainer`)

### Efficiency Measurement
- fvcore (FLOPs/MACs)
- Custom latency/throughput benchmarking

### Environment
- Google Colab (GPU)
- Google Drive (checkpoint + dataset caching)

---

## 🏗️ Architecture

```text
Oxford-IIIT Pet (37 classes)
        ↓
ViT-Small (frozen) + linear probe   →  Teacher
        ↓  (soft labels, temperature-scaled)
   Knowledge Distillation
        ↓
 ┌──────────────┬──────────────┐
 Compact CNN      MobileNetV2      →  Students (baseline vs distilled)
        ↓
 Attention Rollout / Grad-CAM / SHAP
        ↓
 Explanation Agreement (IoU, cosine) + Faithfulness (deletion test)
```

---

## 📁 Project Structure

```text
.
├── distilling_interpretability.ipynb
├── results/
│   ├── phase1_summary_mean_std.csv
│   ├── phase1_summary_table.tex
│   ├── phase2_xai_summary.csv
│   └── phase2_xai_summary_table.tex
├── figures/
│   ├── kd_ablation_heatmap.png / .pdf
│   ├── student_learning_curves_compact_cnn.png / .pdf
│   ├── student_learning_curves_mobilenet_v2.png / .pdf
│   └── four_panel_<arch>_<success|disagreement>_<idx>.png / .pdf
├── checkpoints/            (cached to Google Drive, not versioned in repo)
└── README.md
```

---

## 🚀 Quick Start

### 1) Prerequisites
- Google account (for Colab + Drive caching)
- No local GPU required

### 2) Open in Colab
Open `distilling_interpretability.ipynb` in Google Colab and run cells top to bottom. On first run, the notebook downloads Oxford-IIIT Pet and pretrained ViT-Small weights, and caches both to Google Drive.

### 3) Configuration
Key flags near the top of the notebook:

```python
DATASET = "oxford_pets"     # or "imagenette" as a lighter fallback
SEEDS = [42]                 # extend for a multi-seed run
STUDENT_EPOCHS = 15
```

### 4) Outputs
All tables and figures are written to `results/` and `figures/` and are also cached to a linked Google Drive folder: **[Drive folder link — add here]**

---

## 📊 Results

### Distillation efficiency and accuracy (seed = 42)

| Model | Top-1 | Top-5 | ECE | Params | Compression | Speedup |
|---|---|---|---|---|---|---|
| Teacher (ViT-Small + linear probe) | 88.7% | 99.3% | 0.022 | 21.7M | 1.0x | 1.0x |
| Compact CNN — baseline | 17.7% | 49.7% | 0.040 | 398K | 54.4x | 11.7x |
| Compact CNN — distilled | 16.3% | 48.4% | 0.112 | 398K | 54.4x | 9.8x |
| MobileNetV2 — baseline | 23.6% | 59.8% | 0.236 | 2.27M | 9.5x | 1.36x |
| MobileNetV2 — distilled | 24.0% | 61.8% | 0.301 | 2.27M | 9.5x | 1.44x |

Distillation did not produce a consistent accuracy gain over the from-scratch baseline, and increased calibration error (ECE) for both student architectures.

**Training curves:**

| Compact CNN | MobileNetV2 |
|---|---|
| ![Compact CNN learning curves](figures/student_learning_curves_compact_cnn.png) | ![MobileNetV2 learning curves](figures/student_learning_curves_mobilenet_v2.png) |

**KD hyperparameter ablation (temperature × alpha):**

![KD ablation heatmap](figures/kd_ablation_heatmap.png)

---

### Explainability methods

| Method | Applied to | What it measures |
|---|---|---|
| Attention rollout | Teacher (ViT-Small) | Aggregates transformer attention across all layers into a single per-patch relevance map |
| Grad-CAM | Students (CNN, MobileNetV2) | Uses gradients into the last conv layer to localize regions driving the prediction |
| SHAP (GradientExplainer) | Students | Feature-attribution overlay grounded in input-gradient-based contribution estimates |
| Top-20% region IoU | Teacher vs. student | Overlap between the most-attended regions of each explanation map |
| Cosine similarity | Teacher vs. student | Whole-map structural similarity between explanation heatmaps |
| Deletion faithfulness | Teacher and student, separately | Accuracy drop after masking each model's own top-attended region — higher drop = more faithful explanation |

### Explanation agreement and faithfulness (200 test images per architecture)

| Architecture | Agreement | Top-20% IoU | Cosine sim. | Teacher faithfulness drop | Student faithfulness drop |
|---|---|---|---|---|---|
| Compact CNN | Disagree | 0.126 | 0.724 | 0.229 | -0.079 |
| Compact CNN | Agree | 0.149 | 0.683 | 0.200 | 0.300 |
| MobileNetV2 | Disagree | 0.193 | 0.684 | 0.225 | -0.189 |
| MobileNetV2 | Agree | 0.184 | 0.732 | 0.194 | 0.419 |

Top-20% region overlap between teacher and student explanations is low overall (0.13–0.19), and does not track prediction agreement in the expected direction. Deletion-based faithfulness drops are higher and more consistently positive for the teacher than for either student, and only exceed the teacher's drop for students on agreement cases — suggesting the students' own explanations are often less faithful to their actual decisions than the teacher's.

### Example explanations

| | |
|---|---|
| ![Compact CNN four-panel example](figures/four_panel_compact_cnn_disagreement_0.png) | ![MobileNetV2 four-panel example](figures/four_panel_mobilenet_v2_disagreement_0.png) |

Each panel shows, left to right: original image, teacher attention rollout, student Grad-CAM, student SHAP overlay.

---

## ⚠️ Limitations & Future Work

- Results are reported for a **single seed** (42) due to Colab free-tier session limits; multi-seed variance estimation and significance testing are left for future work.
- The KD hyperparameter ablation is intentionally small to fit free-tier runtime; selected temperature/alpha should be revalidated in a larger sweep.
- Evaluated on a single fine-grained pet dataset — findings may not transfer directly to medical, biological, or underwater/marine domains.
- Attention rollout, Grad-CAM, SHAP, and deletion testing are proxy explanation methods; none guarantees alignment with human expert reasoning, and the teacher itself is not assumed to be perfectly interpretable.
- Planned next steps (pending additional compute): multi-seed runs with statistical significance testing, a wider KD hyperparameter grid, additional teacher/student architectures, and — longer term — extension to domain-specific foundation models with human-expert evaluation of explanation quality.

---

## 📌 Notes

- All checkpoints, cached data, and raw outputs are stored in Google Drive rather than the repo to keep it lightweight: **[Drive folder link — add here]**
- This project was built as a portfolio piece demonstrating self-supervised/foundation-model compression and explainability techniques, intended to be extended into a fuller study.

## 🎥 Demo / Notebook

```md
Notebook: distilling_interpretability.ipynb
```
