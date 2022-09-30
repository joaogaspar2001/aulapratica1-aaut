import numpy as np

x = np.array([[24], [30], [36]])
y = np.array([[13], [14], [16]])

X = np.hstack((np.ones((x.shape[0], 1)), x))
beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

SSE = np.linalg.norm(y - np.matmul(X, beta)) ** 2

new_data = np.array([[1], [34]])
print(np.matmul(new_data.T, beta))