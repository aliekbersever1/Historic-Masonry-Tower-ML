# Masonry Buildings  ML

Machine Learning and Symbolic Regression models for predicting the first and second natural frequencies of historic masonry buildings.

---

## Overview

This repository contains the Python codes developed for the following study:

**Machine Learning and Symbolic Regression Models for Predicting the Natural Frequencies of Masonry Buildings**

The repository includes:

- Group-based Nested Cross Validation
- Hyperparameter Optimization
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Multi-Layer Perceptron (MLP)
- Linear Regression
- Symbolic Regression (PySR)

---

## Repository Structure

```
Masonr-Buildings-ML
│
├── data/
│     Kombinations.xlsx
│
├── src/
│     f1_f2_group_nested_final_test_only_code.py
│     group_based_outer_cv_linear_symbolic_regression_final_fixed.py
│
├── results/
│
├── figures/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Codes

### Machine Learning Models

```bash
cd src
python f1_f2_group_nested_final_test_only_code.py
```

### Linear and Symbolic Regression

```bash
python group_based_outer_cv_linear_symbolic_regression_final_fixed.py
```

---

## Dataset

The dataset used in this repository is located in

```
data/Kombinations.xlsx
```

---

## Outputs

The scripts automatically generate:

- Excel reports
- Word reports
- Prediction tables
- Parity plots
- SHAP plots
- Feature importance plots

All outputs are saved in

```
results/
```

---

## Citation

If you use this repository, please cite the associated publication.

---

## License

This project is distributed under the MIT License.
