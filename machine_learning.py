import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import seaborn
import sklearn
import skimage
import tensorflow as tf
from tensorflow import keras
# Make numpy values easier to read.
#np.set_printoptions(precision=3, suppress=True)
from keras import layers

class MachineLearning:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def wildlife_ml(self):
        train, val, test = np.split(self._data.sample(frac=1), [int(0.8*len(self._data)), int(0.9*len(self._data))])
        print(train)

        batch_size = 5
        train_ds = tf.convert_to_tensor(train)

        [(train_features, label_batch)] = train_ds.take(1)
        print('Every feature:', list(train_features.keys()))
        print('A batch of exporters:', train_features['Exporter'])
        print('A batch of targets:', label_batch )
    
    def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
        # Create a layer that turns strings into integer indices.
        if dtype == 'string':
            index = layers.StringLookup(max_tokens=max_tokens)
        # Otherwise, create a layer that turns integer values into integer indices.
        else:
            index = layers.IntegerLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Encode the integer indices.
        encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

        # Apply multi-hot encoding to the indices. The lambda function captures the
        # layer, so you can use them, or include them in the Keras Functional model later.
        return lambda feature: encoder(index(feature))