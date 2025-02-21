import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data(m, b, num_points=100, noise_std=5):
    """
    Generates random data points (x, y) based on the line y = mx + b with added Gaussian noise.
    """
    x = np.linspace(-10, 10, num_points)
    noise = np.random.normal(0, noise_std, num_points)
    y = m * x + b + noise
    return x, y

def plot_data(x, y, title="Generated Data"):
    """
    Plots the generated data points.
    """
    plt.scatter(x, y, label='Data Points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_regression_line(x, y, m, b, title="Regression Line"):
    """
    Plots the dataset along with the regression line.
    """
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, m * x + b, color='red', label='Regression Line')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()

def solve_linear_regression(X_train, y_train):
    """
    Computes the linear regression coefficients using the closed-form, mathematical solution.
    Students must complete this function using only linear algebra.
    """
    Xb = np.c_[np.ones((len(X_train), 1)), X_train]
    Xb_T = np.transpose(Xb)
    Xb_T_Xb = np.dot(Xb_T, Xb)
    Xb_T_Xb_inv = np.linalg.inv(Xb_T_Xb)
    Xb_T_y = np.dot(Xb_T, y_train)
    b_hat, m_hat = np.dot(Xb_T_Xb_inv, Xb_T_y)

    return m_hat, b_hat  # Return slope and intercept

# Example use case (to be replaced by your script when evaluating the students' code)
if __name__ == "__main__":
    # Generate synthetic data
    m_true, b_true = 3, -2
    x_data, y_data = generate_linear_data(m_true, b_true)
    
    # Split into training and testing sets
    indices = np.random.permutation(len(x_data))
    train_size = int(0.8 * len(x_data))
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    
    X_train, y_train = x_data[train_indices], y_data[train_indices]
    X_test, y_test = x_data[test_indices], y_data[test_indices]
    
    # Solve for linear regression parameters
    m_est, b_est = solve_linear_regression(X_train, y_train)
    
    # Plot the results
    plot_data(x_data, y_data, "Generated Data")
    plot_regression_line(X_test, y_test, m_est, b_est, "Fitted Regression Line")
    
    # Print results
    print(f"Estimated parameters: m = {m_est:.4f}, b = {b_est:.4f}")
