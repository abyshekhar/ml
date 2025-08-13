import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(42)  # For reproducibility
n_samples = 100

# Generate random 2D points such that the x and y axes 
X = np.random.rand(n_samples, 2) * 10  # Points in [0, 10] x [0, 10]
# print(X)
y = (X[:, 1] > X[:, 0]).astype(int)  # Label 1 if y > x, else 0
# print(y)
# Add bias term (append 1 to each input vector)
X_bias = np.c_[X, np.ones(n_samples)]  # Shape: (n_samples, 3)
# print(X_bias)
# Step 2: Perceptron Algorithm
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
    
    def fit(self, X, y):
        # print(X.shape)
        n_features = X.shape[1]
        # print(n_features)
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        # print(f"Self Weights {self.weights}")
        # Training loop
        for _ in range(self.n_iterations):
            for i in range(len(X)):
                # Compute prediction: w * x >= 0 => class 1, else class 0
                prediction = 1 if np.dot(self.weights, X[i]) >= 0 else 0
                # Update weights if misclassified
                if prediction != y[i]:
                    # print(f"prediction {i} actual {y[i]} ")
                    update = self.lr * (y[i] - prediction)
                    # print(f"self.lr * (y[i] - prediction) {update}")
                    self.weights += update * X[i]
                    # print(f"self.weights += update * X[i] =>{self.weights}= {update}* {X[i]}")
    
    def predict(self, X):
        return np.array([1 if np.dot(self.weights, x) >= 0 else 0 for x in X])

# Train the Perceptron
perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_bias, y)

# Step 3: Visualize the results
# Scatter plot of data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0 (below y=x)')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1 (above y=x)')

# Plot the decision boundary: w0*x + w1*y + w2 = 0 => y = -(w0*x + w2)/w1
w = perceptron.weights
x_values = np.array([0, 10])
y_values = -(w[0] * x_values + w[2]) / w[1]
plt.plot(x_values, y_values, 'k-', label='Decision Boundary')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Evaluate accuracy
predictions = perceptron.predict(X_bias)
accuracy = np.mean(predictions == y) * 100
print(f"Accuracy: {accuracy:.2f}%")