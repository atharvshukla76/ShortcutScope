# 🔬 ShortcutScope: Exploring & Mitigating Shortcut Learning in CNNs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## 📊 Benchmark Test Results

| Model Strategy | Overall Accuracy | Worst-Group Accuracy | Macro F1 | G0 (Land/Land) | G1 (Land/Water) | G2 (Water/Land) | G3 (Water/Water) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 55.61% | 3.74% | 0.4844 | 96.98% | 22.35% | 3.74% | 78.97% |
| **Group-Weighted CNN** | 56.14% | 9.35% | 0.4913 | 92.95% | 26.87% | 9.35% | 76.48% |
| **Group-Balanced Sampling CNN** | **55.37%** | **47.54%** | **0.5177** | 58.67% | **47.54%** | **58.26%** | 68.38% |

> **Key Finding**: Group-Balanced Sampling delivers a **~12.7x improvement in worst-group accuracy** (from 3.74% to 47.54%), demonstrating effective shortcut mitigation.

---

## 🌟 Interactive Streamlit Dashboard

ShortcutScope features a custom liquid-glass dashboard (`app.py`) for live diagnosis and interpretability:
- **🧪 Live Test an Image**: Upload any bird image or test with sample birds to get simultaneous predictions from all three models.
- **🔍 Grad-CAM Explainability**: Visual heatmaps revealing where each CNN focuses its attention (background vs. foreground bird anatomy).
- **📈 Model Comparison & Metrics**: Group accuracies, macro F1, and disparity curves.
- **🧩 Confusion Matrices & Classification Reports**: Per-model diagnostic breakdowns.
- **🌗 Liquid Glass Theme Engine**: Interactive toggle switching between Light Coast Glass and Dark Cosmic Obsidian Glass.

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/atharvshukla76/ShortcutScope.git
cd ShortcutScope
```

### 2. Environment Setup
```bash
python -m venv env
# Windows
.\env\Scripts\activate
# Linux/macOS
source env/bin/activate

pip install -r requirements.txt
```

### 3. Launch Interactive Dashboard
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
├── frozen_experiment_2026-10-14/         # Experiment logs and checkpoints (Baseline & Weighted)
├── frozen_experiment_balanced_sampling/  # Experiment logs and checkpoints (Balanced Sampling)
├── train.ipynb                           # Baseline & Weighted model training pipeline
├── balanced_sampling.ipynb               # Group-balanced sampling implementation & training
├── Evaluation.ipynb                      # Comprehensive evaluation and comparative metrics
├── gradcam_analysis.ipynb                # Grad-CAM interpretability and feature extraction
├── robust_analysis.ipynb                 # Robustness and spurious correlation analysis
└── dataset.ipynb                         # Waterbirds dataset exploration & splits
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
