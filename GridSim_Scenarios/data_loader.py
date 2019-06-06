from keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, list_ids, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #  Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples: (n_samples, *dim, n_channels)
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            x[i, ] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]
        return x, y
