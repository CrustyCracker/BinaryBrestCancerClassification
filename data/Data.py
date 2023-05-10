from copy import deepcopy
import pandas as pd
from random import randint


class Data:
    '''
    Data class for the breast cancer dataset.
    self.raw_data: the raw data
    self.raw_x: the raw data without the class attribute
    self.raw_y: the class attribute
    self.preprocessed_data: the preprocessed data
    self.preprocessed_x: the preprocessed data without the class attribute
    self.preprocessed_y: the class attribute
    '''

    def __init__(self, raw_data) -> None:
        self.raw_data = raw_data
        self.raw_x = raw_data.drop('class', axis=1)
        self.raw_y = raw_data['class']
        self._preprocess()

    def _preprocess(self):
        df = deepcopy(self.raw_data)

        # Missing attributes
        attributes_missing = ['node-caps', 'breast-quad']
        for attribute in attributes_missing:
            df[attribute] = df[attribute].replace(
                '?', df[attribute].value_counts().idxmax())

        # One-hot encoding
        attributes_to_onehot = ['menopause', 'breast', 'breast-quad']

        for attribute in attributes_to_onehot:
            new_columns = pd.get_dummies(df[attribute], prefix=attribute)
            df = df.drop(attribute, axis=1)
            df = df.join(new_columns)

        # binary attributes
        binary_attributes = ['node-caps', 'irradiat']

        for attribute in binary_attributes:
            df[attribute] = df[attribute].map(dict(yes=1, no=0))

        # normalize attribtes
        attributes_to_normalize = ['age', 'tumor-size', 'inv-nodes']

        for attribute in attributes_to_normalize:
            df[attribute] = df[attribute].map(lambda a: float(
                a.split('-')[0]) + (float(a.split('-')[1]) -
                                    float(a.split('-')[0])) / 2)
            df[attribute] = (df[attribute]-df[attribute].min()) / \
                (df[attribute].max() - df[attribute].min())

        # class attribute
        classes_dict = {'no-recurrence-events': 0, 'recurrence-events': 1}
        df['class'] = df['class'].map(lambda a: classes_dict[a])

        self.preprocessed_data = df
        self.preprocessed_x = df.drop('class', axis=1)
        self.preprocessed_y = df['class']

    def get_train_and_valid_set(self, frac, seed=None):
        if not seed:
            seed = randint(0, 10000)
        train = self.preprocessed_data.sample(frac=frac,
                                              random_state=seed)
        test = self.preprocessed_data.drop(train.index)

        return train, test
