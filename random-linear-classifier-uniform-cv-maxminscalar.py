import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)  # For reproducibility

num_samples = 100

# Generate dog data: whiskers uniform(4,6), ears uniform(7,9)
dog_whiskers = np.random.uniform(4, 6, num_samples)
dog_ears = np.random.uniform(7, 9, num_samples)
dogs = np.column_stack((dog_whiskers, dog_ears))

# Generate cat data: whiskers uniform(7,9), ears uniform(4,6)
cat_whiskers = np.random.uniform(7, 9, num_samples)
cat_ears = np.random.uniform(4, 6, num_samples)
cats = np.column_stack((cat_whiskers, cat_ears))

# Combine into full dataset
X = np.vstack((dogs, cats))
y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))

# Scale features to [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Shuffle and split into train/test (80/20)
indices = np.arange(X_scaled.shape[0])
np.random.shuffle(indices)
train_size = int(0.8 * X_scaled.shape[0])
train_idx = indices[:train_size]
test_idx = indices[train_size:]
X_train = X_scaled[train_idx]
y_train = y[train_idx]
X_test = X_scaled[test_idx]
y_test = y[test_idx]

# Function to compute misclassification error and return misclassified indices
def compute_error(theta0, theta, X, y, return_misclassified=False):
    preds = np.sign(theta0 + np.dot(X, theta))
    errors = (preds != y)
    error_rate = np.mean(errors)
    if return_misclassified:
        return error_rate, np.where(errors)[0]
    return error_rate

# Function to find best hypothesis for a given K on given data
def find_best_hypothesis(X, y, K, d=2):
    min_error = float('inf')
    best_theta0 = None
    best_theta = None
    for _ in range(K):
        theta = np.random.normal(0, 1, d)
        theta0 = np.random.normal(0, 1)
        error = compute_error(theta0, theta, X, y)
        if error < min_error:
            min_error = error
            best_theta0 = theta0
            best_theta = theta
    return best_theta0, best_theta, min_error

# Cross-validation to tune K
k_values = [50, 100, 200, 500, 1000]  # Increased max K
num_folds = 5
kf = KFold(n_splits=num_folds)

best_k = None
best_cv_error = float('inf')

for k in k_values:
    fold_errors = []
    for train_fold_idx, val_fold_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_fold_idx], X_train[val_fold_idx]
        y_fold_train, y_fold_val = y_train[train_fold_idx], y_train[val_fold_idx]
        
        theta0, theta, _ = find_best_hypothesis(X_fold_train, y_fold_train, k)
        val_error = compute_error(theta0, theta, X_fold_val, y_fold_val)
        fold_errors.append(val_error)
    
    avg_cv_error = np.mean(fold_errors)
    print(f"Average CV error for K={k}: {avg_cv_error}")
    
    if avg_cv_error < best_cv_error:
        best_cv_error = avg_cv_error
        best_k = k

print(f"Best K from CV: {best_k}")

# Retrain on full train with best K
best_theta0, best_theta, train_error = find_best_hypothesis(X_train, y_train, best_k)

# Compute test error and misclassified points
test_error, misclassified_idx = compute_error(best_theta0, best_theta, X_test, y_test, return_misclassified=True)

# Output results
print(f"Best training error: {train_error}")
print(f"Test error: {test_error}")
print(f"Best theta0: {best_theta0}")
print(f"Best theta: {best_theta}")
if len(misclassified_idx) > 0:
    print(f"Misclassified test points (indices in test set): {misclassified_idx}")
    print(f"Misclassified test points (features, labels):")
    for idx in misclassified_idx:
        print(f"  Point {idx}: X={X_test[idx]}, y={y_test[idx]}")
else:
    print("No misclassified test points.")

# Plotting the data and decision boundary (using scaled features)
plt.figure(figsize=(8, 6))

# Scatter plot: dogs (+1) in blue, cats (-1) in red
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='blue', label='Dogs (+1)')
plt.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], color='red', label='Cats (-1)')

# Decision boundary: theta0 + theta[0]*x + theta[1]*y = 0 => y = -(theta0 + theta[0]*x)/theta[1]
x_vals = np.linspace(np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0]), 100)
if best_theta[1] != 0:  # Avoid division by zero
    y_vals = -(best_theta0 + best_theta[0] * x_vals) / best_theta[1]
    plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')
else:
    x_vert = -best_theta0 / best_theta[0]
    plt.axvline(x=x_vert, color='green', label='Decision Boundary')

plt.xlabel('Whisker Length (Scaled)')
plt.ylabel('Ear Flappiness Index (Scaled)')
plt.title('Data Points and Random Linear Classifier Boundary (Uniform Distribution)')
plt.legend()
plt.grid(True)
plt.show()