# predicting-online-news-popularity

This project focuses on predicting Online News Popularity using 4 binary classification algorithms. The task is implemented in PySpark to handle the dataset at scale.

Since news articles are published sequentially, a standard random train/test split would leak future information into training. Instead, we use **rolling window**, which respects the temporal order of the data.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity)

| Property | Value |
|---|---|
| Articles | 39,644 |
| Features | 61 |
| Time span | January 2013 – January 2015 (Mashable) |
| Target | `shares` (continuous) |

### Target Engineering

The raw target (`shares`) is **heavily right-skewed**: most articles receive a modest number of shares while a small fraction go viral (long tail).

**Threshold choice** — using the **median (1,400 shares)** as the binary threshold:

- **1 — Popular:** shares > 1,400
- **0 — Unpopular:** shares ≤ 1,400

This produces a near-balanced split: **49.3% Popular / 50.7% Unpopular**.

---

## Methodology

### Temporal Rolling Window

Because articles are published sequentially, a random split would leak future information into training. Instead, a **fixed-size rolling window** slides forward in time:

| Parameter | Value | Description |
|---|---|---|
| Window size (W) | 10,000 | Rows used for Train + Eval |
| Step size (L) | 1,000 | Window advance per fold = Test block size |
| Train fraction | 0.70 | 70% Train / 30% Eval within the window |
| Total folds | 29 | - |

### Data Leakage Prevention

`StandardScaler` (zero mean, unit variance) is fitted **only on the training window** of each fold and then applied to both train and test sets. This ensures that future information never influences the scaler.

### Feature Engineering

58 numeric features are assembled into a single feature vector using `VectorAssembler`. The columns `url`, `timedelta`, `shares`, and `label` are excluded.

---

## Models & Hyperparameter Grids

Hyperparameters are tuned at every fold using `TrainValidationSplit` (70/30 split, optimising AUC).

| Model | Parameters Tuned | Combinations |
|---|---|---|
| **Logistic Regression** | `regParam` ∈ {0.01, 0.1, 1.0}, `elasticNetParam` ∈ {0.0, 0.5, 1.0} | 9 |
| **Random Forest** | `numTrees` ∈ {10, 50, 100} | 3 |
| **Gradient Boosted Trees** | `maxDepth` ∈ {3, 5}, `maxIter` ∈ {50, 100} | 4 |
| **Linear SVM** | `regParam` ∈ {0.01, 0.1, 1.0}, `maxIter` ∈ {100, 200} | 6 |

---

## Evaluation

Each model is evaluated on the test block of every fold. Final results are the **average across all 29 windows**.

| Metric | Description |
|---|---|
| **AUC** | Area under the ROC curve — threshold-independent separability measure |
| **Accuracy** | Fraction of correctly classified articles |
| **F1-Score** | Weighted harmonic mean of Precision and Recall |
| **Precision** | Of all predicted Popular, how many were actually Popular |
| **Recall** | Of all actual Popular, how many were correctly identified |

---

## Project Structure

```
predicting-online-news-popularity/
│
├── Dataset/
│   └── OnlineNewsPopularity.csv       
│
├── News_popularity_prediction.ipynb   # Main notebook
└── README.md
```

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.14.2 |
| Apache Spark (PySpark) | 4.1.2 |
| NumPy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.10.9 |
| seaborn | 0.13.2 |

Install Python dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

PySpark requires a Java runtime. Install Apache Spark separately or use:

```bash
pip install pyspark==4.1.2
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/predicting-online-news-popularity.git
   cd predicting-online-news-popularity
   ```

2. **Open the notebook**
   ```bash
   jupyter notebook News_popularity_prediction.ipynb
   ```

3. **Run all cells** in order. 

---

## Results

| Model | AUC | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.708038 | 0.655690 | 0.654468 | 0.659649 | 0.655690 |
| Random Forest | 0.714497 | 0.659276 | 0.658670 |  0.663437 | 0.659276 |
| Gradient Boosted Trees | **0.721843** | 0.665207 | 0.664105 |  0.666469 | 0.665207 |
| Support Vector Machine | 0.707126 | 0.653207 | 0.652088 |  0.658951 | 0.653207 |

---

## Hardware Environment

| Component | Specification |
|---|---|
| System | Lenovo IdeaPad Slim 3 15IRH10 (83K1) |
| Processor | Intel Core i7-13620H (13th Gen) — 10 cores / 16 threads |
| RAM | 16 GB |

Spark is configured to use `local[12]` threads with 4 GB driver memory.

