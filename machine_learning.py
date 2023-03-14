from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import tensorflow as tf
import numpy as np
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


class MachineLearning:
    '''
    Class that gets paths for datasets and
    trains a tensorflow model and a decision tree classifier
    sklearn model for each path.
    '''
    def __init__(self, data: pd.DataFrame):
        '''
        Gets a dataframe and hot-key encodes six chosen features
        from the dataframe. Assigns the column 'target' as the
        label
        '''
        self._data = data
        self._features = pd.get_dummies(data[['App.', 'Taxon', 'Exporter',
                                              'Term', 'Purpose', 'Source']])
        self._labels = data['target']

    def tensorflow_ml(self):
        '''
        Method that splits off 20% of the dataset to be the training features
        and labels, then takes the remaining dataset and puts it through
        tensorflow's keras API. Each layer's neuron amount gets smaller
        and smaller until only one neuron remains, which says whether or not
        a given organism will be imported to one of the top 5 biggest
        importer countries.
        '''
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
        model.fit(features_train, labels_train, epochs=5)
        model.evaluate(features_test,  labels_test, verbose=2)

    def decisiontree_ml(self):
        '''
        Method that trains an sklearn decision tree classifier
        model on the original features and labels that were set up
        in the __init__ method. Predicts whether or not
        a given organism will be imported to one of the top 5 biggest
        importer countries.
        Returns accuracy score.
        '''
        model = DecisionTreeClassifier()
        model.fit(self._features, self._labels)
        predictions = model.predict(self._features)
        accuracy = accuracy_score(self._labels, predictions)
        return accuracy
