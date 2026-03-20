# MNIST Classification Experiments with SGD and Random Forest

This repository contains an **exploratory script** that tests various approaches for classifying digits from the **MNIST (Digits) dataset** using **scikit-learn**.  

> ⚠️ **Note:** This code is purely for **testing and learning purposes**. It is not a full, functional model ready for deployment. The goal is to experiment with different classifiers, metrics, thresholds, and error analysis.

---

## 📂 Contents

- `digits.py` – main script with all experiments, comments, and plotting  
- Generated plots:
  - Precision-Recall vs Threshold
  - ROC curves
  - Confusion matrices

---

## 🧪 Experiments and Tests Included

### 1. Data Exploration

- Loaded the **MNIST digits dataset** from `sklearn.datasets`
- Explored individual digits visually using `matplotlib`
- Examined data structure and targets

---

### 2. Train/Test Split

- Initial **manual slicing and shuffling** (commented for demonstration)  
- Proper **stratified split** using `train_test_split` to maintain class distribution  

---

### 3. Binary Classification Example

- Simplified classification on **digit '8' vs all others**
- Classifier: `SGDClassifier`
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Tested **threshold manipulation** to observe effects on predictions
- Plotted **Precision-Recall curves** and **ROC curves**
- Compared **Random Forest** vs **SGDClassifier** on the same binary task  
  - Observed better PR and ROC AUC for Random Forest (~0.9896 vs 0.966)

---

### 4. Multi-Class Classification

- Used **SGDClassifier** with **One-vs-All (OvA)** approach for all 10 digits
- Trained **Random Forest** directly on all classes (native multi-class)
- Demonstrated prediction and probability outputs

---

### 5. Error Analysis

- Scaled data using `StandardScaler` for improved classifier performance
- Created **confusion matrices** for multi-class predictions
- Normalized confusion matrix to compare **error rates per class**
- Set diagonal to 0 to highlight **confusions between classes**
- Visualized using `matplotlib` (grayscale heatmaps)
- Bright cells in the matrix indicate **frequent misclassifications**  
  - Useful to identify **which digits are commonly confused**  

---

## 📊 Libraries Used

- Python 3.12.3  
- `numpy` – numerical computations  
- `matplotlib` – plotting and visualization  
- `scikit-learn` – datasets, classifiers, metrics, preprocessing  

---

## ⚡ Key Takeaways

- SGD is fast but limited as a linear classifier; Random Forest handles MNIST patterns better  
- Binary threshold tuning is critical to balance **precision vs recall**  
- Multi-class error analysis via confusion matrices is very helpful to **identify patterns of misclassification**  
- Many experiments are included for **learning purposes only**

---

