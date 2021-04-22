# # Import required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
# from keras.utils.vis_utils import plot_model
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model, load_model
import seaborn as sns
from scipy import integrate
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# import tqdm

# import datetime
# import os.path


def history_plot_and_save(history, model_name, file_path):

    # Show the history of loss
    plt.figure(figsize=(16, 8))

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Augmented Model Training - {model_name}")
    plt.legend()

    diag_file_name = f'training-{model_name}.png'
    plt.savefig(f'{file_path}{diag_file_name}')

    return plt

def example_plot(eg_win, no_feats, win_size, x, xhat):

    train_plot = x[eg_win]
    predict_plot = xhat[eg_win]
    x_plot = list(range(win_size))
    fig, axes = plt.subplots(no_feats, 1, figsize=(16, 8))

    fig.suptitle(f"{eg_win}th Window Train vs. Predict")

    for sensor in range(no_feats):
        sns.lineplot(x=x_plot, y=train_plot[:,sensor], color='b', ax=axes[sensor], label='x')
        sns.lineplot(x=x_plot, y=predict_plot[:,sensor], color='r', ax=axes[sensor], label='xhat')

    plt.legend()

    return plt

def error_computation(x, xhat, mode='MAE'):
    if mode == 'MAE':
        diff = np.abs(xhat - x)
        error = np.mean(diff, axis=1)
    elif mode == 'Area':
        area_diff = []
        for x_win, xhat_win in zip(x, xhat):

            win_diff = []
            for i in range(x_win.shape[1]):

                x_int = integrate.trapz(x_win[:,i], dx=0.05)
                xhat_int = integrate.trapz(xhat_win[:,i], dx=0.05)
                diff = np.abs(xhat_int - x_int)/(2*x_win.shape[0])
                win_diff.append(diff)
            
            area_diff.append(win_diff)
        error = np.array(area_diff)
    elif mode == 'DTW':
        area_diff = []
        j = 0
        for x_win, xhat_win in zip(x, xhat):

            win_diff = []
            for i in range(x_win.shape[1]):
                x_diff, _ = fastdtw(x_win[:,i], xhat_win[:,i], dist=euclidean)
                win_diff.append(x_diff)
            
            area_diff.append(win_diff)
            j += 1
            if j % 1000 == 0:
                print(f"{j}/{x.shape[0]}")
        error = np.array(area_diff)
    else:
        pass
    return error

def error_hist_and_save(error, mode, model_name, file_path):

    plt.figure(figsize=(16, 8))
    plt.hist(error, bins=30)
    plt.xlabel(mode)
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {mode} - {model_name}")
    
    diag_file_name = f'error-hist-{mode}-{model_name}.png'
    plt.savefig(f'{file_path}{diag_file_name}')

    return plt

def fixed_thresh(pct, error, train_len):
    thresh_err = [pct * np.amax(error)] * train_len #or Define 90% value of max as threshold.
    thresh_err = np.array(thresh_err).flatten()
    return thresh_err


def anomaly_df(train, win_size, error, thresh):
    #Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(train[win_size:])
    anomaly_df['error'] = np.max(error, axis=1)
    anomaly_df['thresh'] = thresh
    anomaly_df['anomaly'] = anomaly_df['error'] > anomaly_df['thresh']

    return anomaly_df


def error_vs_thresh_plot_and_save(df, pct, mode, model_name, file_path):
    #Plot testMAE vs max_trainMAE
    plt.figure(figsize=(16, 8))
    sns.lineplot(x=df['Date'], y=df['error'], label=mode)
    sns.lineplot(x=df['Date'], y=df['thresh'], label='Threshold')
    plt.xlabel("Date")
    if mode == 'MAE':
        plt.ylabel(mode)
        plt.title(f"MAE for Reconstructed Signal - Fixed Threshold={pct} - {model_name}")
    elif mode == 'Area':
        plt.ylabel(mode)
        plt.title(f"Area Difference for Reconstructed Signal - Fixed Threshold={pct} - {model_name}")
    elif mode == 'DTW':
        plt.ylabel(mode)
        plt.title(f"DTW Difference for Reconstructed Signal - Fixed Threshold={pct} - {model_name}")
    else:
        pass
    plt.legend()
    diag_file_name = f'error-thresh-{pct}-{mode}-{model_name}.png'
    plt.savefig(f'{file_path}{diag_file_name}')
    anomalies = df.loc[df['anomaly'] == True]

    return anomalies, plt

def detect_anom_plot_and_save(no_feats, sensors, df, anomalies, pct, mode, model_name, file_path):
    fig, axes = plt.subplots(no_feats, 1, figsize=(16, 16))
    for x in range(no_feats):
        df.plot(kind='line', x='Date', y=sensors[x], ax=axes[x])
        anomalies.plot(kind='scatter', x='Date', y=sensors[x], color='r', ax=axes[x], label='Anomaly')

    fig.suptitle(f"Detected Anomalies - Fixed Threshold={pct} - {mode} - {model_name}")
    plt.legend()
    diag_file_name = f'detected-anomalies-{pct}-{mode}-{model_name}.png'
    plt.savefig(f'{file_path}{diag_file_name}')

