# 🔥 Classification Algorithms From Scratch — Perceptron, LDA & Gaussian Naive Bayes

This repository implements **classic machine learning classifiers from scratch**, without using scikit-learn models.  
The project includes:

- Perceptron classifier  
- Linear Discriminant Analysis (LDA) classifier  
- Gaussian Naive Bayes classifier  
- Custom loss functions  
- Evaluation & visualization scripts  
- A theoretical handwritten PDF with derivations

The classifiers are tested on:

- Linearly separable vs. linearly inseparable datasets  
- Gaussian mixture datasets  
- Decision boundaries, covariances, and accuracy comparisons  
- Perceptron training dynamics

---

## 📁 Project Structure

```
ml-classifiers-from-scratch/
│
├── src/
│   ├── base_estimator.py          # Core estimator interface
│   ├── classifiers.py             # Perceptron, LDA, GaussianNB implementations
│   ├── classifiers_evaluation.py  # Experiment runner + plotting
│   ├── loss_functions.py          # misclassification error + accuracy
│   ├── utils.py                   # Plotly helpers, decision surfaces, ellipses
│   └── __init__.py
│
├── data/
│   ├── gaussian1.npy
│   ├── gaussian2.npy
│   ├── linearly_separable.npy
│   ├── linearly_inseparable.npy
│
├── docs/
│   └── Answers.pdf                # Mathematical derivations & theory
│
├── requirements.txt
└── README.md
```

---

## 🚀 Implemented Models

### 🔹 **BaseEstimator**  
`src/base_estimator.py`

Imitates scikit-learn's API and defines:

- `fit(X, y)`
- `predict(X)`
- `loss(X, y)`
- `fit_predict(X, y)`

It forces every classifier to implement `_fit`, `_predict`, `_loss`.

---

### 🔹 **Perceptron Classifier**  
`src/classifiers.py`

Features:

- Online learning  
- Supports intercept  
- Iterates until convergence or max iterations  
- Callback after every update for tracking loss  
- Works on both separable and inseparable datasets

---

### 🔹 **LDA (Linear Discriminant Analysis)**  
`src/classifiers.py`

Implements:

- Class means  
- Shared covariance matrix  
- Class priors  
- Gaussian likelihood  
- Discriminant functions  
- Full covariance ellipse visualization  

---

### 🔹 **Gaussian Naive Bayes**  
`src/classifiers.py`

Implements:

- Per-class feature means  
- Per-class variances  
- Independence assumption  
- Closed‑form likelihood  
- Diagonal covariance ellipses  

---

## 📊 Evaluation & Visualization

### **1. Perceptron Training Loss**
```bash
python src/classifiers_evaluation.py
```

Generates training loss curves for:

- `linearly_separable.npy`  
- `linearly_inseparable.npy`

---

### **2. LDA vs Gaussian NB Comparison**

Also via:
```bash
python src/classifiers_evaluation.py
```

For:

- `gaussian1.npy`
- `gaussian2.npy`

Includes:

- Two subplots (LDA vs GNB)  
- Predicted class coloring  
- True label marker shapes  
- Class mean markers  
- Covariance ellipses  
- Printed accuracies  

---

## 🧠 Theoretical PDF

`docs/Answers.pdf` includes:

- Likelihood derivations  
- Gaussian / Poisson / Multinomial models  
- MLE estimators  
- LDA discriminant rule  
- Naive Bayes assumptions  
- Perceptron separability theory  

---

## 📦 Installation

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### Perceptron experiment:
```
python src/classifiers_evaluation.py
```

### Gaussian classifier comparison:
```
python src/classifiers_evaluation.py
```

### Import models manually:
```python
from src.classifiers import Perceptron, LDA, GaussianNaiveBayes
```

---

## 🛠 Technologies

- Python  
- NumPy  
- Plotly  
- Matplotlib  

---

## 🎯 Learning Outcomes

- Implement ML models manually  
- Understand generative & discriminative models  
- Visualize classifier behavior  
- Build scikit‑learn–style class architecture  
- Work with Gaussian likelihoods  

---

## 📘 License
MIT License.

---

## 🙌 Author  
**Yair Mahfud**
