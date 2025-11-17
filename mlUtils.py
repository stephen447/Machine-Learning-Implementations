import math

def compute_cost(x, y, w, b):
    """
    Computes the cost (mean squared error) for linear regression.
    
    This function calculates the average squared difference between predicted 
    values and actual target values. The cost function is used to measure 
    how well the model fits the data.
    
    Args:
      x (ndarray (m,)): Data, m examples (feature values)
      y (ndarray (m,)): Target values, m examples (actual values)
      w (scalar): Weight parameter (slope of the line)
      b (scalar): Bias parameter (y-intercept)
      
    Returns:
      total_cost (float): The mean squared error cost
      
    Formula:
      J(w,b) = (1/(2*m)) * Σ(f_wb(x[i]) - y[i])²
      where f_wb(x[i]) = w * x[i] + b
    """
    # Get the number of training examples
    m = x.shape[0] 
    
    # Initialize cost accumulator
    cost = 0
    
    # Loop through all training examples
    for i in range(m):
        # Calculate predicted value: f(x) = w*x + b
        f_wb = w * x.iloc[i] + b
        
        # Accumulate squared error for this example
        cost = cost + (f_wb - y.iloc[i])**2
    
    # Calculate mean squared error (divide by 2m)
    total_cost = 1 / (2 * m) * cost

    return total_cost  


def compute_gradient(x, y, w, b):
    """
    Computes the gradient of the cost function with respect to parameters w and b.
    
    The gradient indicates the direction and magnitude of the steepest increase 
    in the cost function. We use the negative gradient to update parameters 
    during gradient descent.
    
    Args:
      x (ndarray (m,)): Data, m examples (feature values)
      y (ndarray (m,)): Target values, m examples (actual values)
      w (scalar): Current weight parameter
      b (scalar): Current bias parameter
      
    Returns:
      dj_dw (scalar): The gradient of the cost w.r.t. the parameter w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
      
    Formula:
      ∂J/∂w = (1/m) * Σ(f_wb(x[i]) - y[i]) * x[i]
      ∂J/∂b = (1/m) * Σ(f_wb(x[i]) - y[i])
    """
    # Get the number of training examples
    m = x.shape[0]

    # Initialize gradient accumulators
    dj_dw = 0
    dj_db = 0

    # Loop through all training examples
    for i in range(m):
        # Calculate predicted value: f(x) = w*x + b
        y_hat = x.iloc[i] * w + b
        
        # Calculate gradient contribution for this example
        # Gradient w.r.t. w: (prediction - actual) * feature value
        dj_dw_i = (y_hat - y.iloc[i]) * x.iloc[i]
        
        # Gradient w.r.t. b: (prediction - actual)
        dj_db_i = (y_hat - y.iloc[i])
        
        # Accumulate gradients
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    
    # Average the gradients over all examples
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w and b parameters.
    
    Gradient descent is an optimization algorithm that iteratively updates 
    parameters w and b by moving in the direction of the steepest descent 
    (negative gradient) to minimize the cost function.
    
    Args:
      x (ndarray (m,)): Data, m examples (feature values)
      y (ndarray (m,)): Target values, m examples (actual values)
      w_in (scalar): Initial value of parameter w
      b_in (scalar): Initial value of parameter b
      alpha (float): Learning rate (step size for parameter updates)
      num_iters (int): Number of iterations to run gradient descent
      cost_function: Function to compute the cost (e.g., compute_cost)
      gradient_function: Function to compute the gradient (e.g., compute_gradient)
      
    Returns:
      w (scalar): Updated value of parameter w after running gradient descent
      b (scalar): Updated value of parameter b after running gradient descent
      J_history (list): History of cost values at each iteration
      p_history (list): History of parameters [w, b] at each iteration
      
    Algorithm:
      For each iteration:
        1. Compute gradients using gradient_function
        2. Update parameters: w = w - α * ∂J/∂w, b = b - α * ∂J/∂b
        3. Record cost and parameters for analysis
    """
    # Initialize history lists to track progress
    J_history = []
    p_history = []
    
    # Initialize parameters with input values
    w = w_in
    b = b_in
    
    # Perform gradient descent for specified number of iterations
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)     

        # Update Parameters: move in direction opposite to gradient
        # This minimizes the cost function
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration (limit to prevent resource exhaustion)
        if i < 100000:      
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
            
        # Print progress every 10% of iterations (or every iteration if < 10)
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    # Return final parameters and history for graphing/analysis
    return w, b, J_history, p_history

def train(x, y, w_init=0, b_init=0, alpha=0.1, num_iters=1000):
    """
    Train a linear regression model using gradient descent.
    
    This is a convenience function that wraps the gradient_descent function
    with default cost and gradient functions. It provides a simple interface
    for training a linear regression model.
    
    Args:
      x (ndarray (m,)): Training data, m examples (feature values)
      y (ndarray (m,)): Target values, m examples (actual values)
      w_init (scalar): Initial value of parameter w (default: 0)
      b_init (scalar): Initial value of parameter b (default: 0)
      alpha (float): Learning rate - controls step size in gradient descent (default: 0.1)
      num_iters (int): Number of iterations to run gradient descent (default: 1000)
      
    Returns:
      w (scalar): Final value of parameter w (learned weight/slope)
      b (scalar): Final value of parameter b (learned bias/y-intercept)
      J_history (list): History of cost values at each iteration
      p_history (list): History of parameters [w, b] at each iteration
      
    Example:
      >>> import numpy as np
      >>> x = np.array([1, 2, 3, 4, 5])
      >>> y = np.array([2, 4, 6, 8, 10])
      >>> w, b, J_hist, p_hist = train(x, y, alpha=0.01, num_iters=1000)
    """
    # Run gradient descent with compute_cost and compute_gradient functions
    w_final, b_final, J_hist, p_hist = gradient_descent(x, y, w_init, b_init, alpha, 
                                                          num_iters, compute_cost, compute_gradient)
    
    return w_final, b_final, J_hist, p_hist