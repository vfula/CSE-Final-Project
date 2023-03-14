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
from machine_learning import Decision_Tree

def main():
    clean = DataClean('international_trade.csv')
    clean_df = clean.clean_ml()
    # filter for Appendix I species (App. == 1)
    df_app1 = clean_df[clean_df['App.'] == 1]

# group by importer and count the number of occurrences
    grouped = df_app1.groupby('Importer').size().reset_index(name='count')

# sort by count in descending order
    sorted_grouped = grouped.sort_values('count', ascending=False)

# get the importer with the highest count
    top_importer = sorted_grouped.iloc[0]['Importer']
    print(top_importer)
    ml_model = MachineLearning(clean_df)
    dt = Decision_Tree(clean_df)
    accuracy = dt.decisiontree_ml()
    # filter for Appendix I species
    df_app1 = clean_df[clean_df['App.'] == 1]
    # group by importer and count the number of occurrences
    grouped = df_app1.groupby('Importer').size().reset_index(name='count')
    # sort by count in descending order
    sorted_grouped = grouped.sort_values('count', ascending=False)
    # get the importer with the highest count
    top_importer = sorted_grouped.iloc[0]['Importer']


if __name__ == '__main__':
    main()
