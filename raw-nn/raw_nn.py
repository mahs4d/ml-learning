from __future__ import annotations

from typing import Optional, List

import numpy as np
from sklearn.metrics import accuracy_score
from numpy import typing as npt


def sigmoid(z: npt.ArrayLike):
    return 1 / (1 + np.exp(-z))


class InputLayer:
    def __init__(self, layer_size: int):
        self.layer_size = layer_size


class DenseLayer:
    def __init__(self, layer_size: int, prev_layer: Optional[DenseLayer], next_layer: Optional[DenseLayer]):
        self.layer_size = layer_size

        self.prev_layer = prev_layer
        self.next_layer = next_layer

        self.w = np.random.rand(self.prev_layer.layer_size, layer_size)
        self.b = np.random.rand(layer_size, 1)

    def forward_propagate(self, prev_a: npt.ArrayLike, batch_size: int) -> npt.ArrayLike:
        assert prev_a.shape == (self.prev_layer.layer_size, batch_size)

        z = self.w.T.dot(prev_a) + self.b
        assert z.shape == (self.layer_size, batch_size)

        a = sigmoid(z)
        assert a.shape == (self.layer_size, batch_size)

        return a

    def backward_propagate(self, cur_a: npt.ArrayLike, y: npt.ArrayLike, batch_size: int,
                           next_dz: npt.ArrayLike) -> npt.ArrayLike:
        assert cur_a.shape == (self.layer_size, batch_size)

        if self.next_layer is not None:
            g_prime = cur_a * (1 - cur_a)
            dz = (next_dz.T.dot(self.next_layer.w.T)).T * g_prime
        else:
            dz = cur_a - y

        return dz

    def update_parameters(self, cur_dz: npt.ArrayLike, prev_a: npt.ArrayLike, batch_size: int, learning_rate: float):
        dw = (1 / batch_size) * prev_a.dot(cur_dz.T)
        db = (1 / batch_size) * np.sum(cur_dz, axis=1, keepdims=True)

        self.w = self.w - (learning_rate * dw)
        self.b = self.b - (learning_rate * db)


class NeuralNetwork:
    def __init__(self):
        self.layers: List[DenseLayer] = []

        self._layer_sizes = []
        self.input_layer_size = 0

    def set_input_layer_size(self, sz: int):
        self.input_layer_size = sz

    def add_dense_layer(self, layer_size: int):
        self._layer_sizes.append(layer_size)

    def build(self):
        prev_layer = None
        for layer_size in self._layer_sizes:
            d = DenseLayer(layer_size=layer_size,
                           prev_layer=prev_layer if prev_layer else InputLayer(self.input_layer_size),
                           next_layer=None)
            if prev_layer is not None:
                prev_layer.next_layer = d
            prev_layer = d
            self.layers.append(d)

    def train(self, x: npt.ArrayLike, y: npt.ArrayLike, epochs: int = 1000, learning_rate: float = 0.01):
        for e in range(epochs):
            batch_size = x.shape[1]

            # forward propagate all layers and save values
            layers_a = [None] * len(self.layers)
            last_a = x
            for i in range(0, len(self.layers)):
                last_a = self.layers[i].forward_propagate(prev_a=last_a, batch_size=batch_size)
                layers_a[i] = last_a

            # compute error
            loss = (y * np.log(last_a)) + ((1 - y) * np.log(1 - last_a))
            loss = float((1 / batch_size) * np.sum(-loss, axis=1))

            # compute accuracy
            y_pred = last_a.reshape((last_a.shape[1],))
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            accuracy = accuracy_score(y.reshape(y_pred.shape), y_pred=y_pred)

            print(f'epoch {e} > loss: {loss}; accuracy: {accuracy}')



            # backward propagate in reverse order and save values
            layers_dz = [None] * len(self.layers)
            last_dz = None
            for i in range(len(self.layers) - 1, -1, -1):
                last_dz = self.layers[i].backward_propagate(cur_a=layers_a[i], y=y, batch_size=batch_size,
                                                            next_dz=last_dz)
                layers_dz[i] = last_dz

            # update all parameters layer by layer
            for i in range(0, len(self.layers)):
                self.layers[i].update_parameters(
                    cur_dz=layers_dz[i],
                    prev_a=layers_a[i - 1],
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )

    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike:
        batch_size = x.shape[1]
        last_a = x
        for i in range(0, len(self.layers)):
            last_a = self.layers[i].forward_propagate(prev_a=last_a, batch_size=batch_size)

        return last_a
