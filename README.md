# Breast Cancer Prediction

A machine learning project implementing logistic regression for breast cancer prediction. The model uses the Breast Cancer Wisconsin dataset with 30 features to classify tumors as malignant (M) or benign (B).

## Project Overview

This project demonstrates the implementation of a logistic regression model for binary classification of breast cancer tumors. The model achieves **95% accuracy** on the test dataset.

### Dataset

- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Total Samples**: 569
- **Features**: 30 quantitative features (mean, standard error, and worst case values)
- **Classes**: 
  - Malignant (M): 357 samples
  - Benign (B): 212 samples
- **Train-Test Split**: 80-20

## Model Performance

- **Accuracy**: 94.7%
- **Precision**: 93% (class 0), 97% (class 1)
- **Recall**: 99% (class 0), 88% (class 1)
- **F1-Score**: 0.96 (class 0), 0.93 (class 1)

## Technology Stack

- **Python 3**
- **Libraries**:
  - NumPy - Numerical computations
  - Pandas - Data manipulation and analysis
  - Scikit-learn - Machine learning algorithms
  - Seaborn - Data visualization
  - Matplotlib - Plotting and visualization

## Project Structure

```
breast-cancer-prediction/
├── Breast_Cancer.ipynb    # Main Jupyter notebook with model implementation
├── README.md               # Project documentation
├── .gitignore              # Python .gitignore
└── data.csv                # Dataset (if included)
```

## Implementation Details

### Data Preprocessing

1. Load the Breast Cancer Wisconsin dataset
2. Encode target variable: Malignant (M) -> 1, Benign (B) -> 0
3. Separate features and target variable
4. Train-test split (80-20) with stratification

### Model Training

- **Algorithm**: Logistic Regression
- **Solver**: LBFGS
- **Max Iterations**: 1000
- **Random State**: 42 (for reproducibility)

### Model Evaluation

- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization using seaborn heatmap

## Usage

### Running the Notebook

1. Open `Breast_Cancer.ipynb` in Jupyter Notebook
2. Execute cells sequentially from top to bottom
3. Review the model predictions and evaluation metrics

### Requirements

Install the required packages:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Files

- **Breast_Cancer.ipynb**: Complete implementation with data loading, preprocessing, model training, and evaluation
- **README.md**: Project documentation
- **.gitignore**: Git ignore configuration for Python projects

## Key Insights

1. The logistic regression model performs well on this dataset with 95% accuracy
2. The model has high recall for benign tumors (99%) ensuring minimal false negatives
3. Feature scaling and preprocessing are crucial for model performance
4. Class imbalance (357 malignant vs 212 benign) was handled using stratified sampling

## Future Improvements

- Implement additional algorithms (SVM, Random Forest, Neural Networks)
- Perform feature selection and dimensionality reduction
- Hyperparameter tuning using GridSearchCV
- Cross-validation for more robust evaluation
- Feature importance analysis

## Author

Souradeep Howlader

## License

MIT License - Feel free to use this project for educational and research purposes.

## References

- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
