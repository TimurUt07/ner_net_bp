import numpy as np

class NeuralNetwork:
    def __init__(self, layer_neuron_counts, learning_rate):
        self.layer_neuron_counts = layer_neuron_counts
        self.learning_rate = learning_rate
        
        self.weights = [np.random.uniform(size=(layer_neuron_counts[i], layer_neuron_counts[i+1])) for i in range(len(layer_neuron_counts) - 1)]
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def predict(self, input_data):
        layers = [input_data]
        
        for i in range(len(self.weights)):
            layers.append(self.sigmoid(np.dot(layers[i], self.weights[i])))
        
        return layers[-1]
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i in range(len(X)):
                data_point = X[i]
                target = y[i]
                
                layers = [data_point]
                for j in range(len(self.weights)):
                    layers.append(self.sigmoid(np.dot(layers[j], self.weights[j])))
                
                error = target - layers[-1]
                
                deltas = [error * self.sigmoid_derivative(layers[-1])]
                for j in range(len(self.weights)-1, 0, -1):
                    deltas.insert(0, deltas[0].dot(self.weights[j].T) * self.sigmoid_derivative(layers[j]))
                
                for j in range(len(self.weights)):
                    self.weights[j] += np.dot(layers[j].reshape(-1,1), deltas[j].reshape(-1,1).T) * self.learning_rate

# Создание и обучение нейронной сети
neuron_counts = [2, 2, 1]  # Количество нейронов в каждом слое
learning_rate = 0.3 # Скорость обучения

nn = NeuralNetwork(neuron_counts, learning_rate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn.train(X, y, epochs=20000)

# Тестирование нейронной сети
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range(len(X_test)):
    prediction = nn.predict(X_test[i])
    print(f'Входные данные: {list(X_test[i])}, Предсказанный результат: {prediction}')

# Вывод всех весов обученной нейронной сети
all_weights = np.concatenate([w.flatten() for w in nn.weights])
print('Веса в одном массиве:')
print(list(all_weights))


