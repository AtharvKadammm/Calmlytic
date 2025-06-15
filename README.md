# Calmlytic: Anxiety Severity Analysis & Prediction

This project analyzes various lifestyle, physiological, and demographic factors to understand and predict anxiety severity using advanced machine learning techniques.

## 🚀 Project Summary

The aim of Calmlytic is to explore how habits like caffeine intake, sleep, physical activity, heart rate, and stress levels influence anxiety. The project applies various ML models to predict anxiety severity and generate actionable insights.

## 🌐 Project Demo

You can explore the full project visualizations and report here:  
[🔗 Calmlytic Website Demo](https://sites.google.com/view/calmlytic/home)

## 📊 Data Overview

- **Source**: Kaggle — Anxiety Attack Factors, Symptoms, and Severity Dataset.
- **API Used**: Kaggle API
- **Raw Dataset**: Available in `/data/raw_data/`
- **Cleaned Dataset**: Available in `/data/clean_data/`
- **Train-Test Split**: Prepared for different models

Key features include:

- Demographics (Age, Gender, Occupation)
- Lifestyle factors (Caffeine Intake, Sleep Hours, Physical Activity, Alcohol Consumption)
- Physiological measures (Heart Rate, Breathing Rate)
- Stress levels and Anxiety Severity labels

## 🔎 Exploratory Data Analysis (EDA)

Performed extensive EDA to uncover key patterns:

- Correlation Heatmaps
- Stress vs Anxiety Scatterplots
- Sleep patterns by Gender
- Physical Activity distribution
- Caffeine vs Anxiety Severity
- Family History and Anxiety relation
- Occupational Breakdown
- Cluster-based segmentation

## ⚙ Machine Learning Models Used

The following models were implemented:

### Dimensionality Reduction

- PCA (Principal Component Analysis)

### Unsupervised Learning

- K-Means Clustering
- Hierarchical Clustering (Dendrograms)
- DBSCAN

### Association Rule Mining

- Apriori Algorithm

### Supervised Learning

- Naive Bayes (Multinomial, Bernoulli, Gaussian)
- Decision Tree Classifier
- Logistic Regression (binary classification)
- Support Vector Machine (SVM)
- XGBoost (Gradient Boosting Classifier)

### Model Highlights

- **Naive Bayes**: Accuracy ~50% (limited by independence assumption)
- **Decision Tree**: Overfitting risk identified
- **SVM**: Low performance due to overlapping features (~30-34% accuracy)
- **XGBoost**: Best performing model after tuning — identified top anxiety factors with better interpretability.

## 🔑 Key Insights

- High caffeine intake, poor sleep, high physical activity and high stress strongly correlated with severe anxiety.
- Chronic stress consistently emerged as the most dominant factor.
- Lifestyle adjustments can significantly impact anxiety levels.
- A multifactorial combination of physical, behavioral, and psychological factors drive anxiety.
- Personalized interventions are crucial for effective anxiety management.

## 🛠 Technologies & Libraries Used

- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Preprocessing**: sklearn (StandardScaler, LabelEncoder, train_test_split, PCA)
- **Feature Engineering**: imbalanced-learn (SMOTE)
- **Modeling**: sklearn, xgboost, statsmodels
- **Association Rule Mining**: mlxtend
- **Other**: json, jupyter notebook

## 📁 Project Structure

- Calmlytic/
  - CODE-NOTEBOOK/
    - Calmlytic.ipynb
  - DATA/
    - raw_data.csv
    - clean_data.csv
  - TRAIN-TEST-DATA/
    - train_X.csv
    - train_y.csv
    - test_X.csv
    - test_y.csv
  - README.md
  - requirements.txt


## 🌐 Contact

Atharv Kadam

Email: atharva895@gmail.com

---

> *Note: This project was developed as an individual academic machine learning project with end-to-end data preparation, feature engineering, modeling, interpretation, and reporting.*

---

## 🔗 References

- Dataset: https://www.kaggle.com
- API Documentation: https://www.kaggle.com/docs/api
