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


#Decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecisionTreeClassifier:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = data[['App.', 'Taxon', 'Importer', 'Term', 'Purpose', 'Source']]
        self.labels = data['target']

    def decisiontree_ml(self):
        model = DecisionTreeClassifier()
        model.fit(self.features, self.labels)
        predictions = model.predict(self.features)
        accuracy = accuracy_score(self.labels, predictions)
        return accuracy