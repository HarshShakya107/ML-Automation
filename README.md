# ML Explorer Pro

ML Explorer Pro is a web-based machine learning app built using Python and Streamlit. It allows users to explore classification and regression algorithms, visualize results, and evaluate model performance with ease.

## Features

- Upload your dataset or use the built-in Iris dataset.
- Choose from 10 classification and 9 regression algorithms.
- Automatic label encoding for categorical target variables.
- Optional standard scaling for features.
- View key performance metrics:
  - Classification: Accuracy, Confusion Matrix, Classification Report
  - Regression: R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE)
- Visualizations:
  - Decision boundaries for 2-feature classification tasks
  - Predicted vs Actual plots for regression tasks

**Note:** Missing values must be handled before using the app.

## Supported Algorithms

**Classification:**
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Logistic Regression
- SVM Classifier
- Gradient Boosting (GBM)
- XGBoost Classifier
- Naive Bayes
- AdaBoost Classifier
- Extra Trees Classifier

**Regression:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Ridge Regression
- Lasso Regression
- Gradient Boosting Regressor
- XGBoost Regressor
- SVR (Support Vector Regressor)
- KNN Regressor


```bash
pip install -r requirements.txt
