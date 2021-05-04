import pandas as pd
import gnsspy as gp
import georinex as gr
from sklearn import preprocessing
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
        data = raw_data.loc[raw_data['Stacja'] == '352200375WARSZAWA-OKĘCIE']
        data = data.drop(data.columns[[0, 1, 6, 7, 10]], axis=1)
        data = data.loc[(data['Dostępność (Availability).1'] >= 0.99) & (data['Dostępność (Availability)'] >= 0.99)]

        input_data = data.iloc[:,:7]
        output_data = data.iloc[:,-8:]
        print(output_data.tail(1))

        self.input_data = self.normalize_data(input_data)
        self.output_data = output_data
        # print(data[:2], data.columns)
        # print(normalized_data[:2])

    def normalize_data(self, data):
        x = data.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_data = pd.DataFrame(x_scaled, columns=data.columns)
        return normalized_data

    def create_model(self):
        model = keras.Sequential(
            [
                Dense(16, activations.sigmoid, input_shape=(7,)),
                Dense(16, activations.sigmoid),
                Dense(16, activations.sigmoid),
                Dense(16, activations.sigmoid),
                Dense(2, activations.linear),
            ]
        )
        self.model = lm.ModelWrapper(model)
        self.model.compile(optimizer=optimizers.Adam(), loss=lm.MeanSquaredError(),metrics=['accuracy'])

    def train(self):
        validation_input = self.input_data.tail(1)
        print(self.output_data.iloc[:-1].iloc[:, :2])
        self.model.fit(self.input_data.iloc[:-1].to_numpy(), self.output_data.iloc[:-1].iloc[:, :2].to_numpy(), shuffle=True, epochs=100, verbose=2, validation_split=0.2)
        print(self.model.predict(validation_input.to_numpy()))
        print(validation_input)
        pass


def main():
    data_wrapper: DataWrapper = DataWrapper('data_GNSS_EGNOS.csv')
    data_wrapper.create_model()
    data_wrapper.train()
    pass

if __name__ == "__main__":
    main()