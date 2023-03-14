from machine_learning import MachineLearning

import numpy as np
import pandas as pd
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


def main():
    df_big = pd.read_csv('international_trade.csv')
    big_counts = df_big.Importer.value_counts().nlargest(5)
    print(big_counts)

    operate_data('comptab_2023-03-01 23_24_comma_separated.csv')
    operate_data('international_trade.csv')


def clean_ml(path: str):
    '''
    Method that takes a relative path of a csv file dataset
    and adds a column that will be used as the label
    (column is called 'target'). Target will consist
    of ones or zeroes; one means that the importer
    country is one of the top 5 countries that
    import the most animals. Zero means it is not
    a country in the top 5.
    Drops columns that will not be used as features.
    '''
    wildlife = pd.read_csv(path)

    is_PA = wildlife['Importer'] == 'PA'

    wildlife['target'] = np.where(is_PA, 1, 0).astype('int64')

    wildlife = wildlife.drop(columns=['Importer',
                                      'Importer reported quantity',
                                      'Exporter reported quantity', 'Unit',
                                      'Origin', 'Year'])
    print(wildlife.dtypes)

    return wildlife


def operate_data(path: str):
    '''
    Method that cleans the given dataset
    and trains a machine learning model (tensorflow
    or sklearn) on it
    '''
    clean_df = clean_ml(path)
    ml_model = MachineLearning(clean_df)
    ml_model.tensorflow_ml()
    print(ml_model.decisiontree_ml())


if __name__ == '__main__':
    main()
