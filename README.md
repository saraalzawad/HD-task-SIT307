# SIT307 HD Task – Reproducing and Improving a Stacking Ensemble for Heart Attack Prediction

This project reproduces the paper below (Part 1) and then fixes its main weakness (Part 2).

> M. Bhagat, A. Sharma, and P. Agarwal, "An efficient stacking-based ensemble technique for early heart attack prediction," *Multimedia Tools and Applications*, vol. 84, pp. 36351–36375, 2025, doi: 10.1007/s11042-024-19293-7.



---

## 1. Folder contents

```
SIT307_HD/
├── HD_Task_Heart_Stacking.ipynb   # the full project, step by step (Part 1 + Part 2)
├── heart.csv                      # dataset 
├── requirements.txt               # Python packages needed
├── README.md                      # this file
├── figures/                       # created when the notebook runs (4 PNG figures)
└── results/                       # created when the notebook runs (CSV result tables)
```

**Dataset:** Kaggle *Heart Disease Dataset* (Cleveland, Hungary, Switzerland, Long Beach V) – 1025 rows, 13 features + `target` (1 = disease, 0 = no disease).
Source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

---

## 2. Installation

Tested with **Python 3.11** in Jupyter Lab.

**Option A – pip**
```bash
pip install -r requirements.txt
```

**Option B – inside the notebook**
Run the first code cell (`%pip install ...`), then restart the kernel.

> **Mac users:** if `import xgboost` fails with a `libomp` error, run
> `conda install -c conda-forge llvm-openmp` (or `brew install libomp`) and restart the kernel.

---

## 3. How to run

1. Put `HD_Task_Heart_Stacking.ipynb` and `heart.csv` in the same folder.
2. Open the notebook in Jupyter Lab / Jupyter Notebook / VS Code.
3. Run the install cell once and restart the kernel.
4. Click **Run → Run All Cells**.

The whole notebook runs in about **1–3 minutes** (Step 10 is the slowest part).
All figures are saved to `figures/` and all result tables to `results/`.

---

## 4. What each step does

| Step | What it does | Output |
|---|---|---|
| 1 | Load the dataset | shape, class balance |
| 2 | 80/20 split + standardisation (as in the paper) | 820 train / 205 test |
| 3 | Train the six base models (LR, DT, RF, XGB, NB, KNN) | metrics table |
| 4 | Train the 5-fold stacking model (meta-classifier = LR) | metrics table |
| 5 | Compare with the paper's Table 11 | `results/part1_paper_vs_reproduced.csv`, `fig1`, `fig2` |
| 6 | Repeat with 20 random splits (stability) | `results/part1_20_seeds.csv` |
| 7 | Find the limitation: 723 duplicate rows → data leakage | printed counts |
| 8 | Paper pipeline with vs without duplicates (5×5 CV) | `fig3_leakage.png` |
| 9 | Build the proposed model (one-hot + tuning in each fold + probability stacking) | – |
| 10 | Paper method vs proposed method on the same 25 folds + Wilcoxon test | `results/part2_*.csv`, `fig4`, `results/final_overview.csv` |

---

## 5. Expected results

| | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Paper reported (leaky split) | 0.9853 | 1.0000 | 0.9727 | 0.9861 | 0.9880 |
| **Reproduced** (leaky split) | 0.9854 | 1.0000 | 0.9709 | 0.9852 | 1.0000 |
| Paper method (honest 5×5 CV, no duplicates) | 0.8298 | 0.8243 | 0.8781 | 0.8479 | 0.9029 |
| **Proposed method** (honest 5×5 CV, no duplicates) | 0.8344 | 0.8387 | 0.8669 | 0.8496 | 0.9139 |

- **Key finding:** only 302 of 1025 rows are unique. After a random split, 97.1% of test rows also appear in training, so the paper's 98.5% is inflated by leakage.
- **Proposed vs paper method (Wilcoxon, 25 paired folds):** significant improvement in AUC (p = 0.011) and precision (p = 0.023); accuracy, recall and F1 differences are not significant.

---

## 6. Reproducibility notes

- A fixed seed (`SEED = 42`) is used for splitting, cross-validation and all models, so every run gives the same numbers.
- Very small differences (in the 3rd–4th decimal) can happen with different versions of scikit-learn / XGBoost.
- Assumptions (the paper does not report them): random seed, default hyperparameters for base models, Logistic Regression as the meta-classifier.
