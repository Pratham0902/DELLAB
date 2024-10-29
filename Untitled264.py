#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Column names in the dataset:", data.columns)  # Check column names
    
    # Check the data shape
    print("Data shape:", data.shape)
    
    # Extract features and target
    X = data.iloc[:, 1:-1].values  # All columns except the first (Id) and last (Species)
    y = data.iloc[:, -1].values     # The last column as target variable
    
    # Encode the target variable (Species) to numeric values
    le = LabelEncoder()
    y = le.fit_transform(y)  # Converts species names to numerical labels
    
    print("Features shape (X):", X.shape)
    print("Target shape (y):", y.shape)
    
    return X, y

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)   # Initialize weights
    bias = 0              # Initialize bias term
    loss_history = []     # Store loss at each iteration
    
    for i in range(iterations):
        y_pred = np.dot(X, theta) + bias
        error = y_pred - y
        
        # Gradient calculation
        theta_gradient = (1 / m) * np.dot(X.T, error)
        bias_gradient = (1 / m) * np.sum(error)
        
        # Update parameters
        theta -= learning_rate * theta_gradient
        bias -= learning_rate * bias_gradient
        
        # Calculate and store loss
        loss = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(loss)
        
        # Print loss every 100 iterations
        if i % 100 == 0:
            print(f"Gradient Descent Iteration {i}: Loss = {loss}")

    return theta, bias, loss_history

# Stochastic Gradient Descent function
def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0
    loss_history = []  # Store loss at each iteration
    
    for i in range(iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            x_j = X[random_index:random_index+1]
            y_j = y[random_index:random_index+1]
            
            y_pred = np.dot(x_j, theta) + bias
            error = y_pred - y_j
            
            # Stochastic Gradient calculation
            theta_gradient = x_j.T * error
            bias_gradient = error
            
            # Update parameters
            theta -= learning_rate * theta_gradient.flatten()
            bias -= learning_rate * bias_gradient.item()
        
        # Calculate and store loss for the entire dataset
        y_pred_all = np.dot(X, theta) + bias
        loss = (1 / (2 * m)) * np.sum((y_pred_all - y) ** 2)
        loss_history.append(loss)
        
        # Print loss every 100 iterations
        if i % 100 == 0:
            print(f"Stochastic Gradient Descent Iteration {i}: Loss = {loss}")

    return theta, bias, loss_history

# Plotting function for loss history
def plot_loss_history(loss_history_gd, loss_history_sgd):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_gd, label='Gradient Descent', color='blue')
    plt.plot(loss_history_sgd, label='Stochastic Gradient Descent', color='orange')
    plt.title('Loss History Over Epochs')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Usage example
file_path = "C:/Users/Admin/Downloads/iris.csv"  # Ensure this path is correct
X, y = load_data(file_path)

# Run gradient descent
theta_gd, bias_gd, loss_history_gd = gradient_descent(X, y)
print("Parameters from Gradient Descent:", theta_gd, bias_gd)

# Run stochastic gradient descent
theta_sgd, bias_sgd, loss_history_sgd = stochastic_gradient_descent(X, y)
print("Parameters from Stochastic Gradient Descent:", theta_sgd, bias_sgd)

# Plot the loss history
plot_loss_history(loss_history_gd, loss_history_sgd)


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Dataset Info:\n", data.info())
    
    X = data.iloc[:, 1:-1].values  # Features: all columns except ID and target
    y = LabelEncoder().fit_transform(data.iloc[:, -1])  # Encode target directly
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0
    loss_history = []

    for i in range(iterations):
        y_pred = np.dot(X, theta) + bias
        error = y_pred - y
        
        # Parameter update
        theta -= learning_rate * (1 / m) * np.dot(X.T, error)
        bias -= learning_rate * (1 / m) * np.sum(error)
        
        # Track loss
        loss = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(loss)
        
        if i % 100 == 0:
            print(f"GD Iteration {i}: Loss = {loss}")

    return theta, bias, loss_history

# Stochastic Gradient Descent function
def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0
    loss_history = []

    for i in range(iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            x_j = X[random_index:random_index+1]
            y_j = y[random_index:random_index+1]
            
            # Stochastic update
            error = np.dot(x_j, theta) + bias - y_j
            theta -= learning_rate * error * x_j.flatten()
            bias -= learning_rate * error.item()
        
        # Track loss over the full dataset
        loss = (1 / (2 * m)) * np.sum((np.dot(X, theta) + bias - y) ** 2)
        loss_history.append(loss)
        
        if i % 100 == 0:
            print(f"SGD Iteration {i}: Loss = {loss}")

    return theta, bias, loss_history

# Plotting function
def plot_loss_history(loss_history_gd, loss_history_sgd):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_gd, label='Gradient Descent', color='blue')
    plt.plot(loss_history_sgd, label='Stochastic Gradient Descent', color='orange')
    plt.title('Loss History Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
file_path = "C:/Users/Admin/Downloads/iris.csv"
X, y = load_data(file_path)

# Run Gradient Descent
theta_gd, bias_gd, loss_history_gd = gradient_descent(X, y)
print("Parameters from GD:", theta_gd, bias_gd)

# Run Stochastic Gradient Descent
theta_sgd, bias_sgd, loss_history_sgd = stochastic_gradient_descent(X, y)
print("Parameters from SGD:", theta_sgd, bias_sgd)

# Plot the loss history
plot_loss_history(loss_history_gd, loss_history_sgd)


# In[ ]:




