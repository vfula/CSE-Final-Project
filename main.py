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
np.set_printoptions(precision=3, suppress=True)
from keras import layers

from data_cleaning import DataClean
from machine_learning import MachineLearning


def main():
    clean = DataClean('international_trade.csv')
    clean_df = clean.clean_ml()
    ml_model = MachineLearning(clean_df)




if __name__ == '__main__':
    main()
