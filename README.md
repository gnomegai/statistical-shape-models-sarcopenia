# Sarcopenia Detection from Muscle Shape using Statistical Shape Models (SSM)

## 🧠 Context

This project aims to detect **sarcopenia** — the loss of muscle mass and strength due to aging or disease — based on **muscle shape** extracted from **3D ultrasound or MRI** data.
The analysis leverages **Statistical Shape Models (SSM)** to represent and compare muscle shape variations between healthy and sarcopenic subjects.

---

[https://github.com/user-attachments/assets/b9dead79-6232-4799-936e-010229143198](https://github.com/user-attachments/assets/b9dead79-6232-4799-936e-010229143198)

---

## 🧩 Project Structure

```
statistical-shape-models-sarcopenia/
│
├── data/
│   ├── label/                # 3D ultrasound label maps
│   └── correspondences/      # Correspondence points generated using ShapeWorks
│
├── src/
│   ├── create_correspondences.py   # Creation of correspondence points with ShapeWorks
│   ├── shape_analysis.ipynb        # Statistical shape analysis and classification
│   ├── utils.py                    # Utility functions for file and data handling
│   └── Tutorial_Particles.md       # Tutorial for correspondence point generation
│
├── environment.yml
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<user>/statistical-shape-models-sarcopenia.git
cd statistical-shape-models-sarcopenia
```

### 2. Install ShapeWorks

Download and install **ShapeWorks** from the official website:
👉 [https://sciinstitute.github.io/ShapeWorks/latest/install.html](https://sciinstitute.github.io/ShapeWorks/latest/install.html)

ShapeWorks provides a suite of tools for statistical shape modeling, including:

* Extraction of **correspondence points** (particles)
* Generation of **mean shapes**
* Visualization of **shape variation modes**

Ensure the `shapeworks` command is available in your terminal before proceeding.

### 3. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate sarcopenie-ssm
```

---

## 🧠 Methodology

### 0️⃣ Creation of Correspondence Points

* Use **ShapeWorks Groom and Optimize** to align surfaces and generate correspondence points.
* Export particle sets as `.particles` or `.txt` files for subsequent analysis.

### 1️⃣ Mean Shape Analysis

* Load data in **ShapeWorks Studio**.
* Visualize **mean shape** and **principal modes of deformation**.
* Compare mean structures between **healthy** and **sarcopenic** subjects.

### 2️⃣ LDA Separation

* Apply **Linear Discriminant Analysis (LDA)** on correspondence point coordinates.
* Visualize discriminant axes and subject projections to evaluate separation.

### 3️⃣ Anomaly Detection

#### 3a. LDA-based Detection

* Identify outliers based on their Mahalanobis distance from the healthy cluster.

#### 3b. Reconstruction Error

* Compute shape reconstruction errors using PCA or SSM basis.
* Use thresholds to flag potential sarcopenic patterns.

#### 3c. Supervised Classification

* Train a supervised model (e.g., SVM, Random Forest) using labeled correspondence data.
* Evaluate performance using accuracy, F1-score, and confusion matrices.

---

## 🧾 Citation

If you use this work, please cite or acknowledge:

> "Evan Gossard, Statistical Shape Models for Sarcopenia Detection, 2025"

---

## 📧 Contact

For any questions or collaborations, please contact:
**Evan Gossard** — [email protected]
