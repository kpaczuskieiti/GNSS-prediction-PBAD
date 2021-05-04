import pandas as pd
import gnsspy as gp
import georinex as gr
from sklearn import preprocessing

class DataWrapper():
    raw_data: pd.DataFrame

    def __init__(self, file_url: str):
        raw_data = pd.read_csv(file_url)
        data = raw_data.loc[raw_data['Stacja'] == '352200375WARSZAWA-OKĘCIE']
        data = data.drop(data.columns[[0, 1, 6, 7, 10]], axis=1)
        data = data.loc[(data['Dostępność (Availability).1'] >= 0.99) & (data['Dostępność (Availability)'] >= 0.99)]

        input_data = data.iloc[:,:7]
        output_data = data.iloc[:,-8:]

        x = data.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_data = pd.DataFrame(x_scaled, columns=data.columns)
        # print(data[:2], data.columns)
        # print(normalized_data[:2])


def main():
    data_wrapper: DataWrapper = DataWrapper('data_GNSS_EGNOS.csv')
    pass

if __name__ == "__main__":
    main()