import numpy as np
import pandas as pd


def read_data(filename, sep="\t"):
    """
    Reads the ratings from the rating file, given by the filename.
    :param filename: The name of the rating file.
    :param sep: The seperator of the file.
    :return: A Pandas dataframe containing all the ratings.
    """
    col_names = ['user', 'item', 'rate', 'ts']
    df = pd.read_csv(filename, sep=sep, header=0, names=col_names, engine="python")
    df['user'] -= 1
    df['item'] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df['rate'] = df['rate'].astype(np.float32)
    return df


class Batcher:
    """
    Given a dataframe, this Batcher creates an iterable of random batches of the data.
    """
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose([np.array(self.inputs[i]) for i in range(self.num_cols)])

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size, ))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]
