# Import required libraries
import os.path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, RepeatVector,TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Import functions from author defined module
from utils import history_plot_and_save

class LSTMAutoencoder():
    '''A class to aid in the automatic loading or building and training of an LSTM Autoencoder'''

    def __init__(self, layer_one, layer_two, train, patience, model_file, diag_file_path):
        
        # Automatically instantiate variables into class and automatially load or train new model
        self.layer_one = layer_one      # No. of units in first layer
        self.layer_two = layer_two      # No. of units in second layer
        self.train = train              # Training data
        self.patience = patience        # No. of epochs before early stop
        self.model_file = model_file    # File name to save or load model
        self.diag_file_path = diag_file_path    # File path to model
        
        # Check to see if model exists already
        print(f"Checking for {self.model_file}")
        if os.path.isfile(self.model_file):
            # Load it in
            self.load()
        elif self.layer_two is not None:
            # Build new dual layer AE
            self.build_two_layer()
        else:
            # Build new single layer AE
            self.build_one_layer()
        self.model.summary()

        if self.train_req:
            # Train new model with provided data in class initialisation
            self.history = self.train_model()
            # Plot training history
            self.history_plot = history_plot_and_save(self.history, self.model_file, self.diag_file_path)
        else:
            self.history_plot = None

    def build_one_layer(self):
        '''Automatically build new single layer model'''

        # Define Autoencoder architecture
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

        # Define Autoencoder compilation
        self.model.compile(optimizer='adam', loss='mae')
        self.train_req = True

        print("\nSingle layer model built")

    def build_two_layer(self):
        '''Automatically build new double layer model'''

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

        # Define Autoencoder compilation
        self.model.compile(optimizer='adam', loss='mae')
        self.train_req = True

        print("\nDouble layer model built")

    def load(self):
        '''Automatically load in existing model'''
        
        self.model = load_model(self.model_file)
        self.train_req = False

        print(f"\nModel loaded from: {self.model_file}")

    def train_model(self):
        '''Use input data to train the model'''

        print("\nTraining model")

        # Set parameters to train the model
        callback = EarlyStopping(monitor='loss', patience=self.patience, min_delta=1e-3)
        history = self.model.fit(self.train, self.train, epochs=50, batch_size=32, validation_split=0.1, verbose=1, callbacks=[callback])
        # Save the newly trained model to allow for easy reloading and testing
        self.model.save(self.model_file)
        print(f"\nSaved model as: {self.model_file}")

        return history
