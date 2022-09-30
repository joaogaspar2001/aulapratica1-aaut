import numpy as np

# Months (x) and Weight (y)
x = np.array([[24], [30], [36]])
y = np.array([[13], [14], [16]])

# Design matrix
X = np.hstack((np.ones((x.shape[0], 1)), x))

print("===================== Exercise 4. =====================")

# Solution of the normal equations
beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
print("The solution is a = %.2f and b = %.3f" % (beta[1][0], beta[0][0]))

# Cost function - Sum of squared errors
SSE = np.linalg.norm(y - np.matmul(X, beta)) ** 2
print("The associated error is SSE = %.3f" % SSE)

print("\n===================== Exercise 5. =====================")

# New data
x_0 = np.array([[25], [34]])
for i in range(2):
  X_0 = np.array([[1, x_0[i][0]]])
  y_0 = np.matmul(X_0, beta)
  print("For an age of %d, the baby's estimated weight is %.2f" % (x_0[i][0], y_0))