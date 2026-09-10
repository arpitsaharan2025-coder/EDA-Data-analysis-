# 📊 Exploratory Data Analysis (EDA)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

A structured collection of **Exploratory Data Analysis (EDA)** lab assignments and practical work, built in Python. This repository walks through the full data-preprocessing pipeline — cleaning, handling missing values, imputation, discretization, normalization, outlier detection, and visualization — applied across several real-world datasets (COVID-19, wine quality, world happiness, suicide statistics, and employee records).

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Techniques Covered](#-techniques-covered)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Purpose](#-purpose)
- [Author](#-author)
- [Last Updated](#-last-updated)
- [License](#-license)

---

## 🔍 Overview

This repository is a practical, hands-on record of core EDA concepts applied to multiple datasets. Each script is self-contained and focuses on one stage of the data preprocessing/analysis workflow, making it easy to study, reuse, or extend individual techniques.

---

## 📁 Repository Structure

```
EDA-Data-analysis-/
│
├── Experiment 1/                                        # Lab experiment 1 files
├── Experiment 2/                                        # Lab experiment 2 files
├── Experiment 3/                                        # Lab experiment 3 files
├── Experiment 4/                                         # Lab experiment 4 files
│
├── handling_missing_data_25bds0133.py                    # Detecting & handling missing values
├── imputation_tecnhique_25bds0133.py                     # Imputation strategies (mean/median/mode)
├── Discretization_and_Binning_25bds0133.py               # Binning continuous variables
├── Feature_Scaling_and_Normalization_25bds0133.py        # Min-Max / Z-score scaling
├── Outlier_Detection_25bds0133.py                        # Outlier identification & treatment
├── Arpit_Saharan_25bds0133_Exp4_wine_QT.py               # Wine quality EDA
├── covid_data_analysis.py                                # COVID-19 EDA & visualization
├── happiness_analysis.py                                 # World happiness EDA
├── suicide_china_analysis.py                             # Suicide statistics EDA
├── 25bds0133.py / 25bds0133(Moodle).py                   # Assignment submission scripts
│
├── WineQT.csv                                            # Wine quality dataset
├── covid-data.csv                                        # COVID-19 dataset
├── Covid Data(25BDS0133)                                 # Additional COVID data file
├── happiness.csv                                         # World happiness dataset
├── SuicideChina.csv                                       # Suicide statistics dataset
├── employee_dataset.csv                                  # Employee records dataset
├── Iris Code                                             # Dataset taken from Zip file
├── Model Train                                           # Moodle Experiment 5
├── LICENSE                                               # MIT License
└── README.md                                             # Project documentation
```

---

## 🗂️ Datasets

| Dataset | Description |
|---|---|
| `WineQT.csv` | Physicochemical properties and quality ratings of wine samples |
| `covid-data.csv` | COVID-19 case and statistics data |
| `Covid Data(25BDS0133)` | Additional COVID-19 data file |
| `happiness.csv` | World happiness index and related indicators |
| `SuicideChina.csv` | Suicide statistics data (China) |
| `employee_dataset.csv` | Employee records for HR-style analysis |

---

## 🧠 Techniques Covered

| Technique | Script |
|---|---|
| Handling missing data | `handling_missing_data_25bds0133.py` |
| Imputation (mean / median / mode) | `imputation_tecnhique_25bds0133.py` |
| Discretization & binning | `Discretization_and_Binning_25bds0133.py` |
| Feature scaling & normalization | `Feature_Scaling_and_Normalization_25bds0133.py` |
| Outlier detection | `Outlier_Detection_25bds0133.py` |
| Dataset-specific EDA & visualization | `covid_data_analysis.py`, `happiness_analysis.py`, `suicide_china_analysis.py`, `Arpit_Saharan_25bds0133_Exp4_wine_QT.py` |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MedChronicles/EDA-Data-analysis-.git
cd EDA-Data-analysis-
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib
```

---

## ▶️ Usage

Run any script directly with Python:

```bash
python filename.py
```

**Example:**

```bash
python covid_data_analysis.py
```

> **Note:** Most scripts expect their corresponding CSV file (e.g. `WineQT.csv`, `covid-data.csv`, `happiness.csv`) to be present in the same directory. Make sure the relevant dataset is downloaded/available before running a script.

---

## 🎯 Purpose

These scripts were developed as part of **EDA lab/practical assignments** to build a solid, hands-on understanding of:

- Data cleaning and preprocessing
- Handling missing/inconsistent data
- Statistical and visual data analysis
- Preparing data for downstream machine learning tasks

---

## 👤 Author

**Arpit Saharan**

---

## 🕒 Last Updated

**10 September 2026**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
