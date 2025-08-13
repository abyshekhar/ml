import numpy as np

np.random.seed(42)  # For reproducibility

# (Data generation code from Example 1 goes here...)
num_samples = 1600

# Generate dog data: mean (5, 8)
dog_whiskers = np.random.normal(5, 1, num_samples)
# print(f"dogs data after dog_whiskers:{dog_whiskers}")

dog_ears = np.random.normal(8, 1, num_samples)
# print(f"dogs data after dog_ears:{dog_ears}")

dogs = np.column_stack((dog_whiskers, dog_ears))

# print(f"dogs data after column_stack:{dogs}")
# print(f"length of dogs data",len(dogs),f"shape of dogs data :",np.shape(dogs))
# Generate cat data: mean (8, 5)
cat_whiskers = np.random.normal(8, 1, num_samples)
cat_ears = np.random.normal(5, 1, num_samples)
cats = np.column_stack((cat_whiskers, cat_ears))

# Combine into full dataset
X = np.vstack((dogs, cats))
# print(f"X after vstack: {X}")
y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
# print(f"y after hstack: {y}")

# Assuming X and y are already defined as above

# Shuffle and split into train/test (80/20)
indices = np.arange(X.shape[0])
# print(f"Indices before shuffling : {indices}")
np.random.shuffle(indices)
# print(f"Indices after shuffling : {indices}")

xshape = X.shape[0]
# print(f"X.shape[0] :{xshape}")
train_size = int(0.8 * X.shape[0])
# print(f"Train size : {train_size}")
train_idx = indices[:train_size]
# print(f"train_idx = indices[:train_size] =>{train_idx}")
test_idx = indices[train_size:]
# print(f"test_idx = indices[:test_size] =>{test_idx}")

X_train = X[train_idx]
# print(f"X_train = {X_train}")
y_train = y[train_idx]
# print(f"y_train = {y_train}")

X_test = X[test_idx]
# print(f"X_test = {X_test}")

y_test = y[test_idx]
# print(f"X_train = {y_test}")

# (compute_error function from classification.py goes here...)
def compute_error(theta0, theta, X, y):
    preds = np.sign(theta0 + np.dot(X, theta))
    return np.mean(preds != y)

# Parameters (same as before)
K = 100
d = 2

# Initialize
min_error = float('inf')
best_theta0 = None
best_theta = None

# Loop over K random hypotheses (on train data)
for _ in range(K):
    theta = np.random.normal(0, 1, d)
    theta0 = np.random.normal(0, 1)
    error = compute_error(theta0, theta, X_train, y_train)
    if error < min_error:
        min_error = error
        best_theta0 = theta0
        best_theta = theta

# Compute errors
train_error = min_error
test_error = compute_error(best_theta0, best_theta, X_test, y_test)

# Output results
print(f"Sample data dog + cat :{num_samples}")
print(f"Hyperparameter k ={K}")
print(f"Best training error: {train_error}")
print(f"Test error: {test_error}")
print(f"Best theta0: {best_theta0}")
print(f"Best theta: {best_theta}")