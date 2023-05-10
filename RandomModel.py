import numpy as np
import pandas as pd

from BinaryClassificationModel import BinaryClassificationModel


class RandomModel(BinaryClassificationModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'RandomModel'

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return pd.Series(np.random.uniform(0, 1, len(X)))
