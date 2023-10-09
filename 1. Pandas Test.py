import math

import pandas as pd


def sinusoid(value: float):
    return 1 / (1 + math.e**-value)
DataFameCars = pd.DataFrame([["Audi", "BMW", "SOME"], ["Audi1", "BMW1", "SOME1"]])
DataFrameColor = pd.DataFrame([[15, 44, 255], [23, 0, 109]])
print(DataFrameColor / 255)
