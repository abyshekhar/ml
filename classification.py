import numpy as np

np.random.seed(42)  # For reproducibility

num_samples = 100

# Generate dog data: mean (5, 8)
dog_whiskers = np.random.normal(5, 1, num_samples)
print(f"dogs data after dog_whiskers:{dog_whiskers}")

dog_ears = np.random.normal(8, 1, num_samples)
print(f"dogs data after dog_ears:{dog_ears}")

dogs = np.column_stack((dog_whiskers, dog_ears))

print(f"dogs data after column_stack:{dogs}")
print(f"length of dogs data",len(dogs),f"shape of dogs data :",np.shape(dogs))
# Generate cat data: mean (8, 5)
cat_whiskers = np.random.normal(8, 1, num_samples)
cat_ears = np.random.normal(5, 1, num_samples)
cats = np.column_stack((cat_whiskers, cat_ears))

# Combine into full dataset
X = np.vstack((dogs, cats))
print(f"X after vstack: {X}")
y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
print(f"y after hstack: {y}")
# Function to compute misclassification error
def compute_error(theta0, theta, X, y):
    preds = np.sign(theta0 + np.dot(X, theta))
    return np.mean(preds != y)

# Parameters
K = 100  # Number of random hypotheses
d = 2    # Feature dimensions

# Initialize
min_error = float('inf')
best_theta0 = None
best_theta = None

# Loop over K random hypotheses
for _ in range(K):
    theta = np.random.normal(0, 1, d)   # Random weights
    theta0 = np.random.normal(0, 1)     # Random bias
    error = compute_error(theta0, theta, X, y)
    if error < min_error:
        min_error = error
        best_theta0 = theta0
        best_theta = theta

# Output results
print(f"Best training error: {min_error}")
print(f"Best theta0: {best_theta0}")
print(f"Best theta: {best_theta}")