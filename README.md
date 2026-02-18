# K-Nearest Neighbors (KNN) from Scratch & Using Scikit-Learn

## 📌 Overview
This repository demonstrates the implementation and usage of the **K-Nearest Neighbors (KNN)** algorithm for **classification tasks**.

KNN is a **lazy, non-parametric** algorithm that makes predictions based on the similarity between data points using distance metrics. Unlike many machine learning algorithms, KNN does **not build an explicit model** during training; instead, it stores the training data and performs computation at prediction time.

---

## 📦 Project Features
- KNN implementation using **scikit-learn**
- Explanation of how the KNN algorithm works internally
- Train–test split for evaluation
- Model prediction and accuracy measurement
- Clean and minimal code structure

---

## 🧠 How KNN Works
1. Select the number of neighbors (**K**)
2. Calculate the distance between the test point and all training points
3. Identify the **K nearest neighbors**
4. Assign the class using **majority voting**

---

## ⚙️ Technologies Used
- Python 3.x
- NumPy
- Pandas
- Scikit-learn

---

## 📊 Distance Metric
The model uses **Euclidean Distance** by default:

