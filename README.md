<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-informational"></a>
  <a href="https://scikit-learn.org/"><img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.5+-informational"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Made with Jupyter" src="https://img.shields.io/badge/Made%20with-Jupyter-orange.svg">
</p>

# 🧠 Parallel vs Non-Parallel Random Forest Model Evaluation
## 📘 Overview

This project compares parallelization methods in Random Forest Regressors using Python’s multiprocessing.Pool and n_jobs parameter in scikit-learn. The study evaluates model speedup, training efficiency, and predictive accuracy across multiple CPU cores.

## 🧩 Objective

To measure the effect of parallel computation on training, validation, and prediction performance of Random Forest models and identify optimal approaches for time-efficient machine learning workflows.

## ⚙️ Tools & Technologies

- Languages: Python
- Libraries: scikit-learn, multiprocessing, joblib, pandas, NumPy, Matplotlib
- Environment: Jupyter Notebook
- Visualization: Seaborn, Matplotlib
- Report Writing: LaTeX

## 📊 Dataset

[Kaggle Dataset][kaggle-dataset]

[kaggle-dataset]: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos
Contains exercise metrics and calorie burn data used to train and evaluate the Random Forest Regressor.

## 🚀 Key Findings
| Metric             | `n_jobs` Parallelization                     | `Pool` Parallelization |
| ------------------ | -------------------------------------------- | ---------------------- |
| Training Time      | ⬇️ 60% faster (linear scaling up to 8 cores) | ⬇️ ~40% faster         |
| Validation Speedup | Minor improvement                            | No significant gain    |
| Prediction Time    | ~85× faster with `n_jobs`                    | Minimal change         |
| Accuracy (MAE)     | Stable                                       | Stable                 |


## 📈 Visual Results
- Training Time Speedup
- Prediction Time Speedup
- Cross-Validation Comparison


## 🧮 Methodology

- Preprocessing: Cleaned and merged calories.csv and exercise.csv.
- Model Building: Implemented RandomForestRegressor.
- Parallelization: Used n_jobs argument for multi-core execution within scikit-learn. Applied multiprocessing.Pool for manual parallelization.
- Hyperparameter Tuning: Applied GridSearchCV for optimization.
- Benchmarking: Measured training, prediction, and validation times across 1–8 cores.

## 📘 Results Summary

- Parallelization via n_jobs yielded smoother linear scaling with cores.
- Overhead in multiprocessing.Pool limited its benefits on small datasets (~15K records).
- Demonstrated real-world parallel efficiency trade-offs in CPU-bound ML workflows.

## 📚 References
- [Kaggle: fmendes-DAT263x-demos Dataset](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)
- [Scikit-learn: Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Python Multiprocessing Docs](https://docs.python.org/3/library/multiprocessing.html)
- [Joblib Parallel Processing Guide](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
- [Hyperparameter Tuning in Random Forests – Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
