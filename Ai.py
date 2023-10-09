# Algorithm By RoJQ
import math
import typing
import abstract_class
import PIL.Image
import gzip
import struct
import gzip
import struct
import pandas as pd
import numpy as np

# load compressed MNIST gz files and return pandas dataframe of numpy arrays
def load_data(filename, label=False):
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
    return pd.DataFrame(res)


class NeuronConnection(abstract_class.AbstractConnection):
    def __init__(self, main_neuron: abstract_class.AbstractNeuron, sub_neurons: list[abstract_class.AbstractNeuron], weights: list[float]):
        super().__init__(main_neuron, sub_neurons, weights)


class Neuron(abstract_class.AbstractNeuron):
    def __init__(self, active_func: typing.Callable, index_in_site: int, weights: list, index_in_layer: int):
        """
        Данная функция создает нейрон который будет использовать нейроная сеть..
        :param active_func: Функция которой будет активироваться нейрон должна принимать 1 аргумент
        :param index_in_site: Индекс слоя в сети
        :param weights: Веса которые с данным Нейроном связаны (частный случай - входной нейрон)
        :param index_in_layer: Индекс нейрона в слое
        """
        super().__init__(active_func, index_in_site, weights, index_in_layer)

    def set_connection(self, back_layer: abstract_class.AbstractNeuronLayer):
        self.connection = NeuronConnection(self, back_layer.get_neuron_list, self.weights)
        self.connection: NeuronConnection

    def set_layer(self, layer: abstract_class.AbstractNeuronLayer):
        """
        Устанавливает слой для нейрона
        :param layer:
        :return:
        """
        layer.add_neuron(self)


class NeuronOutputLayer(abstract_class.AbstractNeuronLayer):
    def __init__(self, counts_of_results: int):
        super().__init__(activate_function=lambda : "None", count_of_neuron=counts_of_results, index_in_site=-1)

    @classmethod
    def chain_initialization(cls, neuron_count: int, old_layer: abstract_class.AbstractNeuronLayer) -> tuple[abstract_class.AbstractNeuronLayer, int]:
        """
        Данный метод запускает ципную инициализацию слоев.
        :param weight_range:
        :param neuron_count:
        :param index_in_site:
        :param activate_function:
        :param old_layer:
        :return: Возвращает сам слой и его индекс
        """
        new_layer_index = old_layer.index_in_site + 1
        layer = cls(counts_of_results=neuron_count, index_on_site=new_layer_index)
        return layer, new_layer_index


class NeuronInputLayer(abstract_class.AbstractNeuronLayer):
    def __init__(self, count_of_neuron: int, data_input: pd.DataFrame):
        super().__init__(activate_function=lambda: "None", count_of_neuron=count_of_neuron, index_in_site=0)

    @classmethod
    def chain_initialization(cls, neuron_count, index_in_site: int = None, activate_function: typing.Callable = None) -> tuple[abstract_class.AbstractNeuronLayer, int]:
        """
        Данный метод запускает ципную инициализацию слоев.
        :param neuron_count:
        :param index_in_site:
        :param activate_function:
        :return: Возвращает сам слой и его индекс
        """
        layer = cls(count_of_neuron=neuron_count)
        return layer, layer.index_in_site

    def initialize_neurons(self):
        for index_neuron in range(self.count_of_neuron):
            weight = []  # их нет т.к. слой входной и не имеет других весов
            neuron = Neuron(active_func=self.activate_function, index_in_site=self.index_in_site, index_in_layer=index_neuron, weights=weight)
            # here init value
            neuron.set_layer(self)


class NeuronHiddenLayer(abstract_class.AbstractNeuronLayer):
    def __init__(self, weight_range: tuple, index_in_site: int, count_of_neuron: int, activate_function: typing.Callable, old_layer: abstract_class.AbstractNeuronLayer):
        super().__init__( index_in_site = index_in_site, count_of_neuron= count_of_neuron, activate_function= activate_function)

    def __init__random_weights(self):
        pass

    @classmethod
    def chain_initialization(cls, weight_range: tuple[int, int], neuron_count: int, activate_function: typing.Callable, old_layer: abstract_class.AbstractNeuronLayer) -> tuple[abstract_class.AbstractNeuronLayer, int]:
        """
        Данный метод запускает ципную инициализацию слоев.
        :param weight_range:
        :param neuron_count:
        :param index_in_site:
        :param activate_function:
        :param old_layer:
        :return: Возвращает сам слой и его индекс
        """
        new_layer_index = old_layer.index_in_site + 1
        layer = cls(weight_range, new_layer_index, neuron_count, activate_function, old_layer)
        layer.__init__random_weights()
        return layer, new_layer_index


class RqBasicNeuronSite:
    def __init__(self, dataImages: pd.DataFrame, dataLabels: pd.DataFrame, counts_hidden_layers: int):
        self._image_data = dataImages
        self._labels_data = dataLabels
        self.__input_len = self._image_data.count("columns", True)[0]
        self._counts_of_layers = counts_hidden_layers
        self.__weights_range = (-10, 10)
        self.input_layer = NeuronInputLayer(count_of_neuron=self.__input_len, data_input=self._image_data)
        self.hidden_layers = []
        self.output_layer = NeuronOutputLayer(10)

    def unit_next_level(self, neuron_value: pd.DataFrame):
        """
        Определение слудующего слоя
        :param neuron_value:
        :return:
        """
        pass

    def create_layer(self, count_weights: int):
        """
        Создает нейроный слой..
        :param count_weights: Количество весов
        :return:
        """
        pass

    @staticmethod
    def sinusoid(value: float):
        return 1 / (1 + math.e**-value)

    def override_weights(self):
        """
        Переопределние текущих весов...
        :return:
        """
        pass


def pull_data():
    imageData = load_data("ai_tests/train-images-idx3-ubyte.gz")
    labelData = load_data("ai_tests/train-labels-idx1-ubyte.gz", True)
    print(list(imageData.iloc[2]))
    return imageData, labelData

if __name__ == '__main__':
    datas = pull_data()
    RqBasicNeuronSite(datas[0], datas[1], 2)
