import functools
import gzip
import base64
import typing

import pandas as pd
import numpy as np
import struct
VOID_FPOINT = 1000.144# 1000.144  - Число которое сигнализирует о незаполнености нейрона


def load_data(filename, label=False) -> np.ndarray:
    with gzip.open(filename) as gz:
        gz.read(4)
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


class InputData:
    def __init__(self, path_to_image_mnist, path_to_label_mnist):
        self.__image_data = load_data(filename=path_to_image_mnist, label=False)
        self.__label_data = load_data(filename=path_to_label_mnist, label=True)

    @property
    def get_output(self):
        return self.__label_data

    @property
    def get_input(self):
        return self.__image_data


class NeuronSite:
    def __init__(self):
        self.__hidden_layers: list[np.ndarray, ...] = []
        self.__weights_layers: list[np.ndarray, ...] = []
        self.__output_basic: None | np.ndarray = None
        self.input_layer_len = 784

    def generate_neuron_site(self, hidden_layers_count: int, neuron_in_layer: int, max_weights: float, output_layer: np.ndarray):
        previous_layer_length = self.input_layer_len  # Кол-во нейронов в предыдущем слое
        for i in range(hidden_layers_count):
            weights_layers = self.generate_random_weights(max=max_weights, count=(neuron_in_layer * previous_layer_length))
            hidden_layer = np.array([VOID_FPOINT for i in range(neuron_in_layer)])
            previous_layer_length = len(hidden_layer)
            self.__weights_layers.append(weights_layers)
            self.__hidden_layers.append(hidden_layer)

        self.__output_basic = output_layer

    @property
    def weight_layers(self) -> list[np.ndarray, ...]:
        return self.__weights_layers

    @property
    def neuron_layers(self) -> list[np.ndarray, ...]:
        return self.__hidden_layers

    def get_hidden_layer(self, index: int) -> list[np.ndarray, np.ndarray]:
        if index > len(self.__weights_layers):
            return None
        return [self.__hidden_layers[index], self.__weights_layers[index]]

    @property
    def output_layer(self) -> np.ndarray:
        return self.__output_basic

    @staticmethod
    def generate_random_weights(max: float, count: int) -> np.ndarray:
        weights = [np.random.random() * max for i in range(count)]
        weights = np.array(weights)
        return weights

    def reverse_distribution(self, layer_index, neuron_index) -> float:
        pass

    def learn_script(self, required_error_value: float) -> float:
        pass

    def weight_override(self) -> bool:
        pass

    def get_result(self) -> np.ndarray:
        pass

    def shot(self, input_data: np.ndarray) -> np.ndarray:
        pass


def test_1():
    result = NeuronSite.generate_random_weights(5, 10)  # worked
    print(result)


def test_2():  # worked
    output_basic = np.array([VOID_FPOINT for i in range(10)])
    el = NeuronSite()

    before = (el.weight_layers, el.hidden_layers, el.output_layer)
    print("Before")
    for i in range(len(before[0])):
        print(f"Веса: {before[0][i]}, \n Кол-во нейронов: {len(before[0][i])}")
        print(f"Значения нейронов: {before[1][i]}, \n Кол-во нейронов: {len(before[1][i])}")
    else:
        print("Веса, Нейроны - не указаны")
    print(f"Выходной слой: {before[2]}")

    el.generate_neuron_site(4, 5, 7, output_basic)

    after = (el.weight_layers, el.hidden_layers, el.output_layer)
    print("After")
    for i in range(len(after[0])):
        print(f"Веса: {after[0][i]}, \n Кол-во весов: {len(after[0][i])}")
        print(f"Значения нейронов: {after[1][i]}, \n Кол-во нейронов: {len(after[1][i])}")
    print(f"Выходной слой: {after[2]}")


if __name__ == '__main__':
    test_2()
