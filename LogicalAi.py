# load compressed MNIST gz files and return pandas dataframe of numpy arrays
import gzip
import struct
import numpy as np
import pandas as pd


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


def generate_neuron_site(hidden_layers_count: int, neuron_in_layer: int, input_data_len: int) -> list[pd.DataFrame, ...]:
    hidden_layers = []
    back_neuron_layer_count = input_data_len
    for i in range(hidden_layers_count):

        data_weight = [np.random.random() * np.random.choice(np.arange(-5, 5, 1)) for i in range(back_neuron_layer_count * neuron_in_layer)]
        weight_layer = pd.DataFrame(data_weight)
        data_neuron = [0 for i in range(neuron_in_layer)]
        layer_ = pd.DataFrame(data=data_neuron, dtype=pd.Float32Dtype)
        back_neuron_layer_count = neuron_in_layer
        hidden_layers.append(weight_layer)
        hidden_layers.append(layer_)

    return hidden_layers


def get_neuron_site_answer(input_layer: pd.DataFrame, hidden_layers: list[pd.DataFrame], output_layer: pd.DataFrame):
    index_hidden_layer = 0
    while index_hidden_layer != len(hidden_layers):
        WeightLayer = hidden_layers[index_hidden_layer]
        NeuronLayer = hidden_layers[index_hidden_layer + 1]
        if index_hidden_layer == 0:
            back_layer = input_layer.to_numpy()  # Предыдущий слой есть входной
        else:
            back_layer = hidden_layers[index_hidden_layer].to_numpy()  # Предыдущий слой < НЕ РОБИТ

        for i in range(int(NeuronLayer.count())):  # Перебираем индекс каждого нейрона в сети
            print(f"Нейрон с индексом: {i}, В слое: {index_hidden_layer}")
            start_with_index, end_with_index = i * int(len(back_layer)), (i + 1) * int(len(back_layer))
            print(f"Веса выбираются с: {start_with_index}, до: {end_with_index} (приращение = {end_with_index - start_with_index})")
            Weights = WeightLayer[start_with_index: end_with_index]  # <<< wrong code
            ValueOfNeuron = Weights * back_layer
            print(int(Weights.count()), " << Общая длина весов для текущего нейрона")
            print(Weights.max(), "<<< max arg of weight \n")
            print(ValueOfNeuron, "<heere\n")
            NeuronLayer[0][i] = ValueOfNeuron.sum()
            print(NeuronLayer)
            print("\n---\n")






        # end
        index_hidden_layer += 1


def learn_neuron_site(input_data: pd.DataFrame, output_data: pd.DataFrame, output_schema: pd.DataFrame):
    NetworkError = 10  # Начальная ошибка сети
    DataIndex = 0  # Индекс даты начала
    while abs(NetworkError) > 0.05:
        pass


if __name__ == '__main__':
    ImageData = load_data("ai_tests/train-images-idx3-ubyte.gz")
    LabelData = load_data("ai_tests/train-labels-idx1-ubyte.gz", True)

    test_input_layer = ImageData.iloc[0]

    test_input_layer: pd.Series = (test_input_layer / 255)
    test_input_layer = test_input_layer.to_frame()
    test_hidden_layers = generate_neuron_site(2, 5, 784)
    output_layer = None

    answer = get_neuron_site_answer(test_input_layer, test_hidden_layers, output_layer)

