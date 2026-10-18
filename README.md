# 🔬 ShortcutScope: Exploring & Mitigating Shortcut Learning in CNNs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shortcutscope-v5a6gafrglggun8rcas7ys.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Web Application**: Explore the interactive model diagnosis platform directly in your browser: **[ShortcutScope on Streamlit Cloud](https://shortcutscope-v5a6gafrglggun8rcas7ys.streamlit.app/)**

**ShortcutScope** is an empirical deep learning framework and interactive diagnostic platform designed to uncover, evaluate, and mitigate **shortcut learning** and **spurious feature reliance** in Convolutional Neural Networks (CNNs).

Trained and evaluated on the benchmark **Waterbirds** dataset, ShortcutScope investigates how standard models rely on spurious background features (e.g. water vs. land) instead of core semantic features (e.g. bird morphology), and compares mitigation strategies including **Group-Weighted Loss** and **Group-Balanced Sampling**.

---

## 📌 Key Objectives & Methodology

In spurious correlation benchmarks like Waterbirds, classes are strongly confounded by background attributes:
- **Group 0**: Landbird on Land *(Majority / Concordant)*
- **Group 1**: Landbird on Water *(Minority / Spurious Conflict)*
- **Group 2**: Waterbird on Land *(Minority / Spurious Conflict)*
- **Group 3**: Waterbird on Water *(Majority / Concordant)*

Standard Empirical Risk Minimization (ERM) yields models that achieve high overall accuracy by taking the "shortcut" (predicting based on background), but suffer catastrophic failures on minority groups.

### Models Compared
1. **Baseline CNN (Standard Cross-Entropy)**: Optimizes average loss; vulnerable to shortcut exploitation.
2. **Group-Weighted Loss CNN**: Penalizes errors on underrepresented groups using group-frequency weights.
3. **Group-Balanced Sampling CNN**: Enforces uniform sampling across all 4 groups per mini-batch, regularizing feature representations and maximizing worst-group robustness.

---

## 📦 Dataset Information: Waterbirds Benchmark

This project uses the official **Waterbirds** benchmark (`waterbird_complete95_forest2water2`), a canonical dataset specifically designed by [Sagawa et al. (ICLR 2020)](https://arxiv.org/abs/1911.08731) to investigate spurious correlations, shortcut learning, and worst-group generalization in computer vision.

### 🖼️ Dataset Origin & Composition
- **Foreground Objects**: Cropped bird images from the **[Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/)** dataset, categorized into two target classes ($y$):
  - `y = 0`: **Landbird** (e.g., woodpeckers, sparrows, warblers, finches)
  - `y = 1`: **Waterbird** (e.g., albatrosses, pelicans, gulls, ducks)
- **Background Scenes**: Natural landscape images extracted from the **[Places365](http://places2.csail.mit.edu/)** scene recognition dataset:
  - `place = 0`: **Land Background** (e.g., bamboo forest, forest, woods)
  - `place = 1`: **Water Background** (e.g., ocean, lake)
- **Synthetic Composition**: Bird cutouts are combined with land or water scenic backgrounds to synthetically create an artificial, controlled spurious co-occurrence.

### 📊 Spurious Bias & Group Splits
In the training partition, the dataset contains a **95% spurious correlation** between the bird class and the background environment:
- 95% of landbirds appear on land backgrounds, and 95% of waterbirds appear on water backgrounds.
- The 4 distinct evaluation groups ($g = 2y + \text{place}$) are:

| Group ID | Target Label ($y$) | Background ($a$) | Type | Total Images | Train (`split=0`) | Val (`split=1`) | Test (`split=2`) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Group 0** | Landbird (`0`) | Land (`0`) | Majority (Concordant) | 6,220 (52.8%) | 3,498 | 467 | 2,255 |
| **Group 1** | Landbird (`0`) | Water (`1`) | Minority (Spurious Conflict) | 2,905 (24.6%) | 184 | 466 | 2,255 |
| **Group 2** | Waterbird (`1`) | Land (`0`) | Minority (Spurious Conflict) | 831 (7.1%) | 56 | 133 | 642 |
| **Group 3** | Waterbird (`1`) | Water (`1`) | Majority (Concordant) | 1,832 (15.5%) | 1,057 | 133 | 642 |
| **Total** | | | | **11,788** | **4,795** | **1,199** | **5,794** |

> ⚠️ **The Shortcut Vulnerability**: Because minority groups (Group 1 and Group 2) represent only a tiny fraction of the training data (56 and 184 samples, respectively), standard ERM models learn to classify the background rather than the bird anatomy, causing near-total failure on counterfactual test examples (e.g. 3.74% worst-group accuracy).

### 📍 Where the Dataset is Used in This Project
- **Model Training**: Used in [`train.ipynb`](train.ipynb) and [`balanced_sampling.ipynb`](balanced_sampling.ipynb) to train Baseline, Group-Weighted, and Group-Balanced CNNs.
- **Metric Benchmarking**: Used in [`Evaluation.ipynb`](Evaluation.ipynb) and [`robust_analysis.ipynb`](robust_analysis.ipynb) to compute overall, per-group, and worst-group metrics.
- **Explainability**: Used in [`gradcam_analysis.ipynb`](gradcam_analysis.ipynb) to extract Grad-CAM attention heatmaps on conflict test images.
- **Interactive UI**: The Streamlit application ([`app.py`](app.py)) queries `metadata.csv` to load sample test birds and display true labels and group metadata.
- **Local Workspace Path**:
  ```
  archive/
  └── waterbird_complete95_forest2water2/
      ├── metadata.csv
      └── [001.Black_footed_Albatross ... 200.Common_Yellowthroat]/
  ```

### 📥 Where to Download the Dataset
The raw dataset (~1.4 GB compressed) is excluded from this Git repository via `.gitignore` to maintain a lightweight repository. If you wish to reproduce training or run local batch experiments, you can download the dataset from any of the following sources:

1. **Official Stanford Distribution (Recommended)**:
   - **Direct Download Link**: [https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
   - **Official Paper**: Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (ICLR 2020). *"Distributionally Robust Neural Networks for Group Shifts: On the Importance of Worst-Case Losses."* ([arXiv:1911.08731](https://arxiv.org/abs/1911.08731))
   - **Official Repository**: [kohpangwei/group_DRO on GitHub](https://github.com/kohpangwei/group_DRO)
2. **Stanford WILDS Benchmark**:
   - Available as part of the [Stanford WILDS Project](https://wilds.stanford.edu/datasets/#waterbirds).
3. **Kaggle Mirror**:
   - Available under the [Waterbirds Dataset on Kaggle](https://www.kaggle.com/datasets) (search: `waterbird_complete95_forest2water2`).

#### Quick Setup Commands
```bash
# Create archive directory
mkdir -p archive

# Download official tarball
curl -O https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

# Extract to archive directory
tar -xvzf waterbird_complete95_forest2water2.tar.gz -C archive/
```

---

## 📊 Benchmark Test Results

| Model Strategy | Overall Accuracy | Worst-Group Accuracy | Macro F1 | G0 (Land/Land) | G1 (Land/Water) | G2 (Water/Land) | G3 (Water/Water) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 55.61% | 3.74% | 0.4844 | 96.98% | 22.35% | 3.74% | 78.97% |
| **Group-Weighted CNN** | 56.14% | 9.35% | 0.4913 | 92.95% | 26.87% | 9.35% | 76.48% |
| **Group-Balanced Sampling CNN** | **55.37%** | **47.54%** | **0.5177** | 58.67% | **47.54%** | **58.26%** | 68.38% |

> **Key Finding**: Group-Balanced Sampling delivers a **~12.7x improvement in worst-group accuracy** (from 3.74% to 47.54%), demonstrating effective shortcut mitigation.

---

## 🌟 Interactive Streamlit Dashboard

> 🌐 **Live Cloud Demo**: **[Launch ShortcutScope Interactive App](https://shortcutscope-v5a6gafrglggun8rcas7ys.streamlit.app/)** — test images and inspect Grad-CAM heatmaps directly in your browser without any local setup!

ShortcutScope features a custom liquid-glass dashboard (`app.py`) for live diagnosis and interpretability:
- **🧪 Live Test an Image**: Upload any bird image or test with sample birds to get simultaneous predictions from all three models.
- **🔍 Grad-CAM Explainability**: Visual heatmaps revealing where each CNN focuses its attention (background vs. foreground bird anatomy).
- **📈 Model Comparison & Metrics**: Group accuracies, macro F1, and disparity curves.
- **🧩 Confusion Matrices & Classification Reports**: Per-model diagnostic breakdowns.
- **🌗 Liquid Glass Theme Engine**: Interactive toggle switching between Light Coast Glass and Dark Cosmic Obsidian Glass.

---

## 🚀 Getting Started

### 🌐 Option 1: Live Cloud Application (Instant Access)
You can test the models and visualize Grad-CAM heatmaps right away via our deployed cloud app:
- 👉 **[Access ShortcutScope on Streamlit Cloud](https://shortcutscope-v5a6gafrglggun8rcas7ys.streamlit.app/)**

### 💻 Option 2: Run Locally

#### 1. Clone Repository
```bash
git clone https://github.com/atharvshukla76/ShortcutScope.git
cd ShortcutScope
```

#### 2. Environment Setup
```bash
python -m venv env
# Windows
.\env\Scripts\activate
# Linux/macOS
source env/bin/activate

pip install -r requirements.txt
```

#### 3. Launch Local Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501/` in your browser.

---

## 📁 Repository Structure

```
ShortcutScope/
├── app.py                                # Liquid-glass Streamlit interactive web application
├── model.py                              # ShortcutScopeCNN PyTorch architecture definition
├── requirements.txt                      # Project dependencies
├── models/                               # Saved model checkpoint weights (.pt)
│   ├── baseline_cnn.pt
│   ├── weighted_cnn.pt
│   └── balanced_sampling_cnn.pt
├── results/                              # Test metrics, confusion matrices, and training curves
│   ├── all_three_models_test_comparison.csv
│   ├── confusion_matrices.json
│   ├── classification_reports.json
│   └── *.png
├── archive/                              # Local dataset directory (gitignored; see Dataset section)
│   └── waterbird_complete95_forest2water2/
│       ├── metadata.csv
│       └── [bird class folders]/
├── frozen_experiment_2026-10-14/         # Experiment logs and checkpoints (Baseline & Weighted)
├── frozen_experiment_balanced_sampling/  # Experiment logs and checkpoints (Balanced Sampling)
├── train.ipynb                           # Baseline & Weighted model training pipeline
├── balanced_sampling.ipynb               # Group-balanced sampling implementation & training
├── Evaluation.ipynb                      # Comprehensive evaluation and comparative metrics
├── gradcam_analysis.ipynb                # Grad-CAM interpretability and feature extraction
├── robust_analysis.ipynb                 # Robustness and spurious correlation analysis
├── dataset.ipynb                         # Waterbirds dataset exploration & splits
└── DataVerify.ipynb                      # Dataset split integrity & image verification
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
