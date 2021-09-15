import numpy as np

# x = np.array([-3, -2, -1, 0, 1, 2, 3])
# y = np.array([-2, -1, 0, 1, 2])
# x, y = np.meshgrid(x, y)
# vector = np.stack([x.flat, y.flat], axis=1)

x, y = np.ogrid[-5:5 + 1, -3:3 + 1]
vector = np.array([x, y])


sigma = [[1, 1], [1, 1]]
sigma = np.array(sigma)

h = np.exp(-0.5 * np.dot(np.dot(vector, sigma), vector.T))

print(h)
