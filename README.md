# Linear Regression Project

A machine learning project that implements linear regression from scratch using gradient descent to predict car prices based on vehicle features, specifically focusing on the relationship between mileage and price.

## Overview

This project demonstrates:

- **Custom implementation** of linear regression using gradient descent
- **Comparison** with scikit-learn's LinearRegression
- **Visualization** of the relationship between mileage and car prices
- **Cost function optimization** through iterative gradient descent

## Dataset

The project uses `dataTestwithout0.csv`, which contains car listing data with the following features:

- `make`: Car manufacturer
- `model`: Car model
- `year`: Manufacturing year
- `mileage`: Vehicle mileage
- `fuelType`: Type of fuel (e.g., Diesel Plug-in Hybrid, Petrol Plug-in Hybrid)
- `engineSize`: Engine size
- `price`: Target variable (car price)
- Additional features: `sellerType`, `transmission`, `color`, `condition`, `owners`, etc.

## Features

### Custom Implementation

- **Cost Function**: Mean squared error (MSE) implementation
- **Gradient Computation**: Manual calculation of gradients for weight (w) and bias (b)
- **Gradient Descent**: Iterative optimization algorithm with configurable learning rate and iterations
- **Data Normalization**: Feature normalization for better convergence

### Visualization

- Scatter plots showing the relationship between mileage and price
- Cost function convergence over iterations
- Linear regression fit visualization

### Comparison

- Side-by-side comparison with scikit-learn's LinearRegression to validate the custom implementation

## Requirements

The project uses the following Python libraries:

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning library (for comparison)

## Installation

1. Clone or download this repository
2. Ensure you have Python 3.12+ installed
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install required packages:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook "Linear Regression-Copy1.ipynb"
   ```

2. Run the cells sequentially to:
   - Load and explore the data
   - Implement gradient descent from scratch
   - Visualize the results
   - Compare with scikit-learn's implementation

## Key Functions

### `compute_cost(x, y, w, b)`

Calculates the mean squared error cost function for linear regression.

### `compute_gradient(x, y, w, b)`

Computes the gradients of the cost function with respect to parameters w and b.

### `gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function)`

Performs gradient descent optimization to find optimal parameters w and b.

## Results

The model learns a linear relationship between mileage and car price:

- **Weight (w)**: Approximately -0.405 (indicating negative correlation: higher mileage = lower price)
- **Bias (b)**: Approximately 85,105 (base price when mileage is 0)

The custom implementation produces results that match scikit-learn's LinearRegression, validating the correctness of the implementation.

## Project Structure

```
ML/
├── README.md                          # This file
├── Linear Regression-Copy1.ipynb      # Main notebook with implementation
├── Linear Regression.ipynb            # Alternative notebook
├── dataTestwithout0.csv              # Dataset
└── venv/                              # Virtual environment
```

## Notes

- The dataset has been preprocessed to remove entries with 0 values (hence "without0" in filename)
- The implementation focuses on a single feature (mileage) for simplicity, but the code structure supports multi-feature regression
- Learning rate and iteration count can be adjusted in the gradient descent settings

## Future Improvements

- Extend to multiple features (multi-variate linear regression)
- Add feature engineering and preprocessing steps
- Implement regularization (Ridge/Lasso regression)
- Add model evaluation metrics (R², RMSE, MAE)
- Cross-validation for better model assessment
