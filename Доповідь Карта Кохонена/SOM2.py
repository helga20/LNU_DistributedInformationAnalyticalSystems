import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SOM3D:
    def __init__(self, shape, input_dim):
        self.shape = shape
        self.weights = np.random.rand(*shape, input_dim)

    def train(self, data, epochs=100, learning_rate=0.1, sigma=1.0):
        for epoch in range(epochs):
            for x in data:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, learning_rate, sigma)
            sigma *= 0.99  # Зменшуємо розмір сигмоїди з кожною епохою
            learning_rate *= 0.99  # Зменшуємо швидкість навчання з кожною епохою

    def find_bmu(self, x):
        dists = np.linalg.norm(self.weights - x, axis=-1)
        return np.unravel_index(np.argmin(dists), self.shape)

    def update_weights(self, x, bmu, learning_rate, sigma):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    distance = np.linalg.norm(np.array([i, j, k]) - np.array(bmu))
                    influence = np.exp(-distance**2 / (2 * sigma**2))
                    self.weights[i, j, k] += learning_rate * influence * (x - self.weights[i, j, k])

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    ax.scatter(i, j, k, color=self.weights[i, j, k] / 255)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

# Приклад використання
data = np.random.rand(100, 3)  # Приклади для навчання
shape = (5, 5, 5)  # Розмір тривимірної сітки
input_dim = 3  # Розмірність вхідних даних

som = SOM3D(shape, input_dim)
som.train(data, epochs=100)
som.visualize()
