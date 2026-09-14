# 🏡 House Prices: Advanced EDA, PCA & Clustering Analysis

An end-to-end Data Science project analyzing residential home features, cleaning complex missing data, normalizing skewed distributions, reducing dimensionality via Principal Component Analysis (PCA), and segmenting properties using K-Means Clustering.

---

## 📌 Project Overview
The goal of this project is to explore, preprocess, and discover underlying patterns in housing data using unsupervised and statistical techniques. 

### Key Highlights from the Analysis:
- **Dataset Exploration:** Initial analysis of 1,460 training instances and 81 features.
- **Outlier Handling:** Detected and removed 2 severe outliers with `GrLivArea > 4,000 sq ft` and `SalePrice < $300,000` to prevent model distortion.
- **Target Transformation:** Addressed right-skewed target variable (`SalePrice`) using log-transformation (`np.log1p`), achieving a near-normal distribution.
- **Missing Value Imputation:**
  - Categorical features with missing structural data (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `GarageType`, etc.) imputed with `"None"`.
  - Numerical features (`GarageArea`, `TotalBsmtSF`, etc.) imputed with `0`.
  - `LotFrontage` imputed using the median of each respective `Neighborhood`.
  - Remaining low-frequency missing values imputed with the mode.
- **One-Hot Encoding:** Categorical variables were converted into binary features, expanding the dataset to 261 columns.
- **Dimensionality Reduction (PCA):** Reduced 258 scaled feature columns to **139 Principal Components** while retaining **90.22% of total variance**.
- **Unsupervised Segmentation (K-Means):**
  - Selected key structural dimensions (`GrLivArea`, `GarageArea`, `TotalBsmtSF`, `LotArea`, `GarageCars`, `FullBath`, `TotRmsAbvGrd`).
  - Determined optimal cluster count ($k=3$) using the **Elbow Method** and validated via **Silhouette Analysis** ($0.2779$).

---

## 🛠️ Tech Stack & Libraries
- **Python 3**
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)

---

## 📂 Project Structure
```text
├── Houses.ipynb          # Interactive Jupyter Notebook with all code & plots
├── train.csv             # Training dataset
├── test.csv              # Test dataset
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
