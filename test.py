import pandas as pd
import gnsspy as gp
import georinex as gr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import levenberg_marquardt.levenberg_marquardt as lm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, optimizers, losses
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from main import DataWrapper

def main():
    data_wrapper: DataWrapper = DataWrapper('data_GNSS_EGNOS.csv')
    data_wrapper.model = keras.models.load_model('Adam-128-mae-mae-relu-3')
    result = data_wrapper.model.predict(data_wrapper.input_data_test)

    error = result - data_wrapper.output_data_test.iloc[:, :1].to_numpy()
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [HNSE]')
    _ = plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    main()