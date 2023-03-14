from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import tensorflow as tf
import numpy as np
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


class MachineLearning:
    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._features = pd.get_dummies(data[['App.', 'Taxon', 'Exporter',
                                              'Term', 'Purpose', 'Source']])
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
                      loss=tf.keras.losses.BinaryCrossentropy(
                           from_logits=True),
                      metrics=['accuracy'])
        model.fit(features_train, labels_train, epochs=1)
        model.evaluate(features_test,  labels_test, verbose=2)

    def decisiontree_ml(self):
        model = DecisionTreeClassifier()
        model.fit(self._features, self._labels)
        predictions = model.predict(self._features)
        accuracy = accuracy_score(self._labels, predictions)
        return accuracy
