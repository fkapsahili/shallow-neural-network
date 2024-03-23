import math
import inspect
from dataclasses import dataclass

import numpy as np


@dataclass
class ShallowNeuralNetworkConfig:
    input_size: int = 2
    output_size: int = 2
    hidden_size: int = 4
    learning_rate: float = 0.01


class ShallowNeuralNetwork:
    def __init__(self, config: ShallowNeuralNetworkConfig):
        assert config.input_size > 0
        assert config.hidden_size > 0
        self.config = config

        self.weights = self._init_weights()
        self.bias = self._init_bias()

    def _init_weights(self):
        self.W1 = np.random.randn(
            self.config.hidden_size, self.config.input_size) * 0.01
        self.W2 = np.random.randn(
            self.config.output_size, self.config.hidden_size) * 0.01

    def _init_bias(self):
        self.b1 = np.zeros((self.config.hidden_size, 1))
        self.b2 = np.zeros((self.config.output_size, 1))

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = np.tanh(self.Z1)
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))

        # Validate the shape of the output
        assert (self.A2.shape == (self.config.output_size, X.shape[1]))
        return self.A2

    def compute_loss(self, A2, Y):
        m = Y.shape[1]  # number of examples
        loss = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        return loss

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.A2 - Y
        dW2 = 1/m * np.dot(dZ2, self.A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(self.A1, 2))
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.config.learning_rate * dW1
        self.b1 -= self.config.learning_rate * db1
        self.W2 -= self.config.learning_rate * dW2
        self.b2 -= self.config.learning_rate * db2

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.config})"
