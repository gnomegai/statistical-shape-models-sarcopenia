# Sarcopenia Detection from Muscle Shape using Statistical Shape Models (SSM)

## 🧠 Context

This project aims to detect **sarcopenia** (loss of muscle mass and strength due to aging or disease) based on **muscle shape** extracted from **3D ultrasound or MRI** data.
The analysis relies on **Statistical Shape Models (SSM)** to represent and compare muscle shape variations between healthy and sarcopenic subjects.

---

https://github.com/user-attachments/assets/b9dead79-6232-4799-936e-010229143198


## 🧩 Project Structure

```
sarcopenie-ssm/
│
├── data/
│   ├── label/               # 3D ultrasound label maps
│   └── correspondences/      # Correspondence points generated using ShapeWorks
│
├── src/
│   ├── create_correspondences.py   # Creation of correspondence points with ShapeWorks
│   └── shape_analysis.ipynb        # Statistical shape analysis and classification
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<user>/sarcopenie-ssm.git
cd sarcopenie-ssm
```

### 2. Install ShapeWorks

Download and install **ShapeWorks** from the official website:
👉 [https://sciinstitute.github.io/ShapeWorks/latest/install.html](https://sciinstitute.github.io/ShapeWorks/latest/install.html)

ShapeWorks provides a suite of tools for statistical shape modeling, including:

* extraction of **correspondence points** (particles),
* generation of **mean shapes**,
* and interactive visualization of shape variations.

Ensure that the `shapeworks` command is available in your terminal before proceeding.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Methodology

### 0️⃣ Correspondence Point Generation

* Use **ShapeWorks Groom and Optimize** to align surfaces and generate correspondence points.
* Export particle sets as `.particles` or `.txt` files for later analysis.

### 1️⃣ Mean Shape Analysis

* Open the data in **ShapeWorks Studio**.
* Visualize mean shapes and principal deformation modes.
* Compare healthy and sarcopenic groups.

### 2️⃣ LDA Separation

* Apply **Linear Discriminant Analysis (LDA)** on the correspondence coordinates to separate the two groups.
* Visualize discriminant axes and subject projections.

### 3️⃣ Anomaly Detection

#### 3a. LDA-based Detection

* Detect outliers based on the distance to the healthy group in LDA space.

#### 3b. Reconstruction Error

* Reconstruct shapes from the first SSM modes.
* Use **reconstruction error** as an indicator of abnormal deformation.

#### 3c. Supervised Classification

* Train a supervised model (e.g., logistic regression) using SSM coefficients or LDA projections.
* Evaluate performance using accuracy, AUC, sensitivity, and specificity.

---

## 📊 Expected Results

* Visualization of **mean shapes** and **main variation modes** between groups.
* Clear separation between healthy and sarcopenic subjects using LDA.
* Reliable anomaly detection scores based on reconstruction error and/or classification.

---

## 👤 Author

**Evan Gossard**
Electrical engineering student specializing in signal and image processing.
Research project on **sarcopenia detection from muscle shape** using **Statistical Shape Models**.

