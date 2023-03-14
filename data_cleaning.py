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

class DataClean:
    def __init__(self, data: str):
        self._data = data

    def clean_ml(self):
        wildlife = pd.read_csv(self._data)

        wildlife['target'] = np.where(wildlife['Importer'] == 'PA', 1, 0).astype('int64')

        #wildlife_features = wildlife_train.copy()
        #wildlife_labels = wildlife_features.pop('target')
        wildlife = wildlife.drop(columns=['Importer','Importer reported quantity', 'Exporter reported quantity', 'Unit', 'Origin', 'Year'])
        print(wildlife.dtypes)

        return wildlife