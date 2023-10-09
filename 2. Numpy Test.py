import math

import pandas as pd
import numpy as np

random_weights_choose = lambda: np.random.random() * np.random.choice(np.arange(-10, 10, 1))

random_list = [round(np.random.random(), 3) for i in range(784)]
random_weights = [round(random_weights_choose(), 2) for i in range(5)]

pd_frame = pd.DataFrame(random_list, dtype=pd.Float64Dtype)
np_frame = pd_frame.to_numpy(dtype=pd.Float64Dtype)
weights_array = np.array(random_weights)
neuron_value = []
for weight_index, weight_value in enumerate(weights_array):
    print(weight_value)
    weight_value_new = weight_value * np_frame
    weight_value_new = weight_value_new.sum()
    print(type(weight_value_new), "\n")
    neuron_value.append(weight_value_new)
neuron_value = np.array(neuron_value)

print(neuron_value.sum(), "<<< Значение нейрона")

print(f"Фрейм pd:\n {pd_frame} \n")

print(f"Фрейм np:\n {np_frame} \n")

print(f"Frame of random weights: \n {weights_array} \n")

print(f"New vector after mult:\n {new_vector} \n")

