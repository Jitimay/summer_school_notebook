"""
DSA 2026 Entry Assessment - Section 3: Machine Learning & Visualization
Solutions for Questions M1 to M4
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# =============================================================================
# QUESTION M1 — Load Data and Create Scatterplot
# =============================================================================

data = np.load('data/data-2class.npz')

d = data['d']
l = data['l']

l_flat = l.flatten()

plt.figure(figsize=(8, 6))
plt.scatter(d[l_flat == 0, 0], d[l_flat == 0, 1], color='red',  alpha=0.5, label='Class 0')
plt.scatter(d[l_flat == 1, 0], d[l_flat == 1, 1], color='blue', alpha=0.5, label='Class 1')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Scatterplot of Data Points by Class')
plt.legend(['Class 0', 'Class 1'])
plt.show()


# =============================================================================
# QUESTION M2 — Draw a Separating Line
# =============================================================================

plt.figure(figsize=(8, 6))
plt.scatter(d[l_flat == 0, 0], d[l_flat == 0, 1], color='red',  alpha=0.5, label='Class 0')
plt.scatter(d[l_flat == 1, 0], d[l_flat == 1, 1], color='blue', alpha=0.5, label='Class 1')

x_line = np.linspace(d[:, 0].min(), d[:, 0].max(), 100)
y_line = x_line * 0 + 0      # Horizontal line at y=0 — adjust slope/intercept as needed

plt.plot(x_line, y_line, color='green', linewidth=2, label='Separating Line')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Scatterplot with Separating Line')
plt.legend(['Class 0', 'Class 1', 'Separating Line'])
plt.show()


# =============================================================================
# QUESTION M3 — Fit Gaussian Distributions
# =============================================================================

points_class_0 = d[l_flat == 0]
points_class_1 = d[l_flat == 1]

mean_class_0 = np.mean(points_class_0, axis=0)
mean_class_1 = np.mean(points_class_1, axis=0)

cov_class_0 = np.cov(points_class_0, rowvar=False)
cov_class_1 = np.cov(points_class_1, rowvar=False)

print("Class 0:")
print(f"  Mean: {mean_class_0}")
print(f"  Covariance:\n{cov_class_0}")

print("\nClass 1:")
print(f"  Mean: {mean_class_1}")
print(f"  Covariance:\n{cov_class_1}")


# =============================================================================
# QUESTION M4 — Heatmap with Gaussian Contours
# =============================================================================

x_min, x_max = d[:, 0].min() - 1, d[:, 0].max() + 1
y_min, y_max = d[:, 1].min() - 1, d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

gaussian_class_0 = multivariate_normal(mean=mean_class_0, cov=cov_class_0)
gaussian_class_1 = multivariate_normal(mean=mean_class_1, cov=cov_class_1)

pos = np.dstack((xx, yy))
pdf_class_0 = gaussian_class_0.pdf(pos)
pdf_class_1 = gaussian_class_1.pdf(pos)

plt.figure(figsize=(10, 7))
cf = plt.contourf(xx, yy, pdf_class_0 + pdf_class_1, levels=20, cmap='viridis', alpha=0.7)
plt.contour(xx, yy, pdf_class_0, colors='red',  levels=5, linewidths=1.5)
plt.contour(xx, yy, pdf_class_1, colors='blue', levels=5, linewidths=1.5)

plt.scatter(d[l_flat == 0, 0], d[l_flat == 0, 1], color='red',  alpha=0.4, s=15, label='Class 0')
plt.scatter(d[l_flat == 1, 0], d[l_flat == 1, 1], color='blue', alpha=0.4, s=15, label='Class 1')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Distributions Heatmap with Data Points')
plt.colorbar(cf, label='Probability Density')
plt.legend()
plt.show()
