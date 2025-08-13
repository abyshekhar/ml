import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde

np.random.seed(42)  # For reproducibility

num_samples = 100

# Generate dog data: whiskers ~ Exp(2) (shorter), ears ~ Exp(4) (flappier)
dog_whiskers = np.random.exponential(scale=2, size=num_samples)
dog_ears = np.random.exponential(scale=4, size=num_samples)
dogs = np.column_stack((dog_whiskers, dog_ears))

# Generate cat data: whiskers ~ Exp(4) (longer), ears ~ Exp(2) (less floppy)
cat_whiskers = np.random.exponential(scale=4, size=num_samples)
cat_ears = np.random.exponential(scale=2, size=num_samples)
cats = np.column_stack((cat_whiskers, cat_ears))

# Combine into full dataset
X = np.vstack((dogs, cats))
y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))

# Print summary statistics to verify distributions
print("Dogs Whiskers (Exp(2)): mean=%.2f, min=%.2f, max=%.2f" % 
      (np.mean(dog_whiskers), np.min(dog_whiskers), np.max(dog_whiskers)))
print("Cats Whiskers (Exp(4)): mean=%.2f, min=%.2f, max=%.2f" % 
      (np.mean(cat_whiskers), np.min(cat_whiskers), np.max(cat_whiskers)))
print("Dogs Ears (Exp(4)): mean=%.2f, min=%.2f, max=%.2f" % 
      (np.mean(dog_ears), np.min(dog_ears), np.max(dog_ears)))
print("Cats Ears (Exp(2)): mean=%.2f, min=%.2f, max=%.2f" % 
      (np.mean(cat_ears), np.min(cat_ears), np.max(cat_ears)))

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
k_values = [50, 100, 200, 500, 1000]
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

# Create figure with five subplots
fig = plt.figure(figsize=(24, 10))

# Subplot 1: Scatter plot with decision boundary (scaled features)
ax1 = fig.add_subplot(231)
ax1.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='blue', label='Dogs (+1)')
ax1.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], color='red', label='Cats (-1)')
x_vals = np.linspace(np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0]), 100)
if best_theta[1] != 0:
    y_vals = -(best_theta0 + best_theta[0] * x_vals) / best_theta[1]
    ax1.plot(x_vals, y_vals, color='green', label='Decision Boundary')
else:
    x_vert = -best_theta0 / best_theta[0]
    ax1.axvline(x=x_vert, color='green', label='Decision Boundary')
ax1.set_xlabel('Whisker Length (Scaled)')
ax1.set_ylabel('Ear Flappiness Index (Scaled)')
ax1.set_title('Data Points and Decision Boundary')
ax1.legend()
ax1.grid(True)

# Subplot 2: Whisker Length PDFs (raw data)
ax2 = fig.add_subplot(232)
x_range = np.linspace(0, np.max(X[:, 0]), 100)
ax2.hist(dog_whiskers, bins=20, density=True, alpha=0.5, color='blue', label='Dogs (Hist)')
ax2.plot(x_range, 0.5 * np.exp(-0.5 * x_range), 'b-', label='Dogs Exp(2)')
ax2.hist(cat_whiskers, bins=20, density=True, alpha=0.5, color='red', label='Cats (Hist)')
ax2.plot(x_range, 0.25 * np.exp(-0.25 * x_range), 'r-', label='Cats Exp(4)')
ax2.set_xlabel('Whisker Length (Raw)')
ax2.set_ylabel('Density')
ax2.set_title('Whisker Length Distributions')
ax2.legend()
ax2.grid(True)

# Subplot 3: Ear Flappiness Index PDFs (raw data)
ax3 = fig.add_subplot(233)
x_range = np.linspace(0, np.max(X[:, 1]), 100)
ax3.hist(dog_ears, bins=20, density=True, alpha=0.5, color='blue', label='Dogs (Hist)')
ax3.plot(x_range, 0.25 * np.exp(-0.25 * x_range), 'b-', label='Dogs Exp(4)')
ax3.hist(cat_ears, bins=20, density=True, alpha=0.5, color='red', label='Cats (Hist)')
ax3.plot(x_range, 0.5 * np.exp(-0.5 * x_range), 'r-', label='Cats Exp(2)')
ax3.set_xlabel('Ear Flappiness Index (Raw)')
ax3.set_ylabel('Density')
ax3.set_title('Ear Flappiness Index Distributions')
ax3.legend()
ax3.grid(True)

# Subplot 4: Dogs 2D KDE with scatter (raw data)
ax4 = fig.add_subplot(234)
x_grid = np.linspace(0, np.max(X[:, 0]), 100)
y_grid = np.linspace(0, np.max(X[:, 1]), 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
kde_dogs = gaussian_kde(dogs.T)
Z_dogs = np.reshape(kde_dogs(positions).T, X_grid.shape)
ax4.contourf(X_grid, Y_grid, Z_dogs, cmap='Blues', alpha=0.5)
ax4.scatter(dogs[:, 0], dogs[:, 1], color='blue', s=10, label='Dogs')
ax4.set_xlabel('Whisker Length (Raw)')
ax4.set_ylabel('Ear Flappiness Index (Raw)')
ax4.set_title('Dogs 2D KDE (Exp(2), Exp(4))')
ax4.legend()
ax4.grid(True)

# Subplot 5: Cats 2D KDE with scatter (raw data)
ax5 = fig.add_subplot(235)
kde_cats = gaussian_kde(cats.T)
Z_cats = np.reshape(kde_cats(positions).T, X_grid.shape)
ax5.contourf(X_grid, Y_grid, Z_cats, cmap='Reds', alpha=0.5)
ax5.scatter(cats[:, 0], cats[:, 1], color='red', s=10, label='Cats')
ax5.set_xlabel('Whisker Length (Raw)')
ax5.set_ylabel('Ear Flappiness Index (Raw)')
ax5.set_title('Cats 2D KDE (Exp(4), Exp(2))')
ax5.legend()
ax5.grid(True)

plt.tight_layout()
plt.show()