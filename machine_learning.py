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
        self._features = data[['App.', 'Taxon', 'Importer', 'Term', 'Purpose', 'Source']]
        self._labels = data['target']

    def tensorflow_ml(self):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self._features, self._labels, test_size=0.2)
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(features_train, labels_train, epochs=1)
        model.evaluate(features_test,  labels_test, verbose=2)
