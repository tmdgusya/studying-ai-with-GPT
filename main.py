import numpy as np

# Example data: house sizes (in square feet)
house_sizes = np.array([750.0, 1000.0, 1250.0, 1500.0, 1750.0])

# Actual house prices (in dollars)
actual_prices = np.array([100000.0, 150000.0, 200000.0, 250000.0, 300000.0])

# Random initial guesses for house prices
initial_guesses = np.array([120000.0, 180000.0, 220000.0, 260000.0, 320000.0])

# Function to calculate mean squared error (MSE) loss
# Loss function will be using as a helper function that helps minimize to gap between actual values and predicted values
def calculate_mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Initial MSE loss
initial_loss = calculate_mse_loss(initial_guesses, actual_prices)
print("Initial MSE loss:", initial_loss)

# Gradient descent to minimize MSE loss
learning_rate = 0.0001
num_iterations = 10000

# Update guesses iteratively to minimize loss
current_guesses = initial_guesses.copy()
for i in range(num_iterations):
    # Predicted prices based on current guesses
    predictions = current_guesses
    
    # Calculate gradients (2 * (predictions - targets))
    gradients = 2 * (predictions - actual_prices)
    
    print(gradients, current_guesses)
    
    # Update guesses using gradients and learning rate
    current_guesses -= learning_rate * gradients
    
    # Calculate MSE loss
    current_loss = calculate_mse_loss(current_guesses, actual_prices)
    
    # Print progress
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}: MSE loss = {current_loss:.2f}")

# Final predicted prices and loss
final_loss = calculate_mse_loss(current_guesses, actual_prices)
print("\nFinal predicted prices:", current_guesses)
print("Final MSE loss:", final_loss)
