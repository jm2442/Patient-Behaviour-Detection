# # Import required libraries
# import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# from keras.callbacks import EarlyStopping
# from keras.utils.vis_utils import plot_model
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model, load_model
# import seaborn as sns
# from scipy import integrate
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean

import os.path
from utils import history_plot_and_save

# import tqdm

# import datetime

class AutoEncoder():
    def __init__(self, layer_one, layer_two, train, model_file, callback, diag_file_path):

        self.layer_one = layer_one
        self.layer_two = layer_two
        self.train = train
        self.model_file = model_file
        self.callback = callback
        self.diag_file_path = diag_file_path

        print(f"Checking for {self.model_file}")

        if os.path.isfile(self.model_file):
            self.load()
        elif self.layer_two is not None:
            self.build_two_layer()
        else:
            self.build_one_layer()
        self.model.summary()

        if self.train_req:
            self.history = self.train_model()
            self.history_plot = history_plot_and_save(self.history, self.model_file, self.diag_file_path)
        else:
            self.history_plot = None

    def build_one_layer(self):
        print("\nSingle layer model built")
        # Define AutoEncoder architecture
        self.model = Sequential()
        # Encoder
        self.model.add(LSTM(self.layer_one, input_shape=(self.train.shape[1], self.train.shape[2]), return_sequences=False))
        self.model.add(Dropout(rate=0.2))
        # Bridge
        self.model.add(RepeatVector(self.train.shape[1]))
        #Decoder
        self.model.add(LSTM(self.layer_one, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.train.shape[2])))
        self.model.compile(optimizer='adam', loss='mae')
        self.train_req = True

    def build_two_layer(self):
        print("\nDouble layer model built")
        # Define AutoEncoder architecture
        self.model = Sequential()
        # Encoder
        self.model.add(LSTM(self.layer_one, input_shape=(self.train.shape[1], self.train.shape[2]), return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(LSTM(self.layer_two, return_sequences=False))
        # Bridge
        self.model.add(RepeatVector(self.train.shape[1]))
        #Decoder
        self.model.add(LSTM(self.layer_two, return_sequences=True))
        self.model.add(LSTM(self.layer_one, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.train.shape[2])))
        self.model.compile(optimizer='adam', loss='mae')
        self.train_req = True

    def load(self):
        print(f"\nModel loaded from: {self.model_file}")
        self.model = load_model(self.model_file)
        self.train_req = False

    def train_model(self):
        print("\nTraining model")
        history = self.model.fit(self.train, self.train, epochs=50, batch_size=32, validation_split=0.1, verbose=1, callbacks=[self.callback])

        self.model.save(self.model_file)
        print(f"\nSaved model as: {self.model_file}")

        return history

    def geez(self):
        print('my guy')

