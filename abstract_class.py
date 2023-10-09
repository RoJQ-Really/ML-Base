import typing


class AbstractNeuron:
    def __init__(self, active_func: typing.Callable, index_in_site: int, weights: list, index_in_layer: int):
        """
        Данная функция создает нейрон который будет использовать нейроная сеть..
        :param active_func: Функция которой будет активироваться нейрон должна принимать 1 аргумент
        :param index_in_site: Индекс слоя в сети
        :param weights: Веса которые с данным Нейроном связаны (частный случай - входной нейрон)
        :param index_in_layer: Индекс нейрона в слое
        """
        self.value = 0  # value - of a neuron
        self.weights = weights
        self.activation_func = active_func  # activation - fucn
        self.index_in_site = index_in_site  # index in site
        self.index_in_layer = index_in_layer  # index in layer
        self.connection = "None"


class AbstractNeuronLayer:
    def __init__(self, index_in_site: int, count_of_neuron: int, activate_function: typing.Callable):
        self.activate_function = activate_function
        self.index_in_site = index_in_site
        self.count_of_neuron = count_of_neuron
        self.__list_of_neurons = []
        print("Создан нейрон!")

    @property
    def get_neuron_list(self) -> list[AbstractNeuron]:
        return self.__list_of_neurons

    def add_neuron(self, v: AbstractNeuron):
        v.index_in_layer = len(self.__list_of_neurons)
        self.__list_of_neurons.append(v)

    def initialize_neurons(self):
        """overwrite - this function - for future"""
        pass



class AbstractConnection:
    def __init__(self, main_neuron: AbstractNeuron, sub_neurons: list[AbstractNeuron], weights: list[float]):
        if len(sub_neurons) != len(weights):
            raise AttributeError("Количество весов не совпадает с количеством под-нейронов!")
        self.main_neuron = main_neuron
        self.sub_neurons = sub_neurons
        self.weights = weights