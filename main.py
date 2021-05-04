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

class DataWrapper():
    raw_data: pd.DataFrame
    data: pd.DataFrame
    input_data: pd.DataFrame
    output_data: pd.DataFrame

    def __init__(self, file_url: str):
        raw_data = pd.read_csv(file_url)
        data = raw_data
        data = data.drop(data.columns[[0, 1, 6, 7, 10]], axis=1)
        data = data.loc[(data['Dostępność (Availability).1'] >= 0.99) & (data['Dostępność (Availability)'] >= 0.99)]

        self.input_data_train = self.normalize_data(data.iloc[:,:7])
        self.output_data_train = data.iloc[:,-8:]
        self.input_data_test = self.normalize_data(data.iloc[:,:7])
        self.output_data_test = data.iloc[:,-4:]

        # self.input_data = self.normalize_data(input_data)
        # self.output_data = output_data
        # print(data[:2], data.columns)
        # print(normalized_data[:2])

    def normalize_data(self, data):
        x = data.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_data = pd.DataFrame(x_scaled, columns=data.columns)
        return normalized_data

    def create_model(self):
        self.model = keras.Sequential(
            [
                Dense(128, activations.relu, input_shape=(7,)),
                Dense(128, activations.relu),
                Dense(128, activations.relu),
                Dense(128, activations.relu),
                Dense(128, activations.relu),
                Dense(1, activation=activations.linear),
            ]
        )
        self.model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])

    def train(self):
        train_output = (self.output_data_train.iloc[:, :1])
        print(self.input_data_train)
        self.model.fit(self.input_data_train.to_numpy(), train_output.to_numpy(), validation_data=(self.input_data_test.to_numpy(), (self.output_data_test.iloc[:, :1]).to_numpy()), shuffle=True, epochs=400, verbose=2)
        print(self.model.predict(self.input_data_test.to_numpy()))
        print(self.output_data_test.iloc[:, :1])
        pass


def main():
    data_wrapper: DataWrapper = DataWrapper('data_GNSS_EGNOS.csv')
    data_wrapper.create_model()
    data_wrapper.train()
    pass

if __name__ == "__main__":
    main()