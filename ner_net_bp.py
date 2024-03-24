import numpy as np

class NeuralNetwork:
    def __init__(self, layer_neuron_counts, learning_rate, weights=None):
        self.layer_neuron_counts = layer_neuron_counts
        self.learning_rate = learning_rate
        
        if weights is not None:
            self.weights = self.unpack_weights(weights)
        else:
            self.weights = [np.random.uniform(size=(layer_neuron_counts[i], layer_neuron_counts[i+1])).flatten() for i in range(len(layer_neuron_counts) - 1)]
        
    def unpack_weights(self, weights):
        unpacked_weights = []
        start = 0
        for count in range(len(self.layer_neuron_counts) - 1):
            end = start + self.layer_neuron_counts[count] * self.layer_neuron_counts[count+1]
            unpacked_weights.append(weights[start:end].reshape(self.layer_neuron_counts[count], self.layer_neuron_counts[count+1]))
            start = end
        return unpacked_weights
    
    def pack_weights(self):
        packed_weights = np.concatenate([w.flatten() for w in self.weights])
        return packed_weights
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def predict(self, input_data):
        layers = [input_data]
        
        for i in range(len(self.weights)):
            layers.append(self.sigmoid(np.dot(layers[i], self.weights[i])))
        
        return layers[-1]

# Введите веса обученной нейронной сети в виде массива
trained_weights = np.array([0.934855210890053, 7.520853399556634, 0.9342385294734243,
7.51945266072536, -29.49913176387905, 23.483867843619848])

# Создание экземпляра нейронной сети с введенными весами
neuron_counts = [2, 2, 1]  # Количество нейронов в каждом слое
learning_rate = 0.3
nn = NeuralNetwork(neuron_counts, learning_rate, weights=trained_weights)

# Пример использования нейронной сети
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range(len(X_test)):
    prediction = nn.predict(X_test[i])
    print(f'Входные данные: {list(X_test[i])}, Предсказанный результат: {prediction}')



