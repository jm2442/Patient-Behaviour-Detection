# # Import required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import integrate
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
# import tqdm

import datetime
import os.path


def history_plot_and_save(history, model_name, file_path, ylimits=None, save_new=True):

    # Show the history of loss
    plt.figure(figsize=(5, 8))

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if ylimits is not None:
        plt.ylim(ylimits)
    plt.title(f"Model Training - {model_name}")
    plt.legend()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    diag_file_name = f'training-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'training-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f'Saved: {full_path}')

    return plt

def example_plot(eg_win, no_feats, win_size, x, xhat, model_name, file_path, docplot=2, save_on=True):

    train_plot = x[eg_win]
    predict_plot = xhat[eg_win]
    x_plot = list(range(win_size))

    if docplot == 1:
        plot_size = (16,8)
    elif docplot == 2:
        plot_size = (8,8)
    elif docplot == 3:
        plot_size = (5,8)
    else:
        plot_size = (5,8)

    fig, axes = plt.subplots(no_feats, 1, figsize=plot_size, sharey=True, sharex=True)

    fig.suptitle(f"{eg_win}th Window - Original vs. Reconstructed Signal - {model_name}")

    for sensor in range(no_feats):
        sns.lineplot(x=x_plot, y=train_plot[:,sensor], color='b', ax=axes[sensor], label='x')
        sns.lineplot(x=x_plot, y=predict_plot[:,sensor], color='r', ax=axes[sensor], label='xhat')


    axes[0].set_ylabel("Norm. Back Angle")
    axes[0].grid(True, which='both')

    axes[1].set_ylabel("Norm. Left Angle")
    axes[1].grid(True, which='both')

    axes[2].set_ylabel("Norm. Right Angle")
    axes[2].grid(True, which='both')
    plt.ylim((0,1))

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.legend()

    diag_file_name = f'eg-window-{eg_win}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    if save_on:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'eg-window-{eg_win}-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
 
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

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
        error = None
    return error

def error_hist_and_save(error, mode, model_name, file_path, save_on=True):

    plt.figure(figsize=(16, 8))
    plt.hist(error, bins=30)
    plt.xlabel(mode)
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {mode} - {model_name}")
    
    diag_file_name = f'error-hist-{mode}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    if save_on:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'error-hist-{mode}-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
    
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return plt


def sig_fig_round(number, sig_fig):
    rounded_number = round(number, sig_fig - int(math.floor(math.log10(abs(number)))) - 1)
    return rounded_number

def multiple_error_hist_plot_and_save(errors, modes, model_name, file_path, label_ypos, traintest, pct=None, save_new=True):

    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 8))
    labels = ['Back Error', 'Left Error', 'Right Error']
    thresholds = []
    x = 0
    for error, mode in zip(errors, modes):
        y, _, _ = axes[x].hist(error, bins=30, label=labels)
        axes[x].grid(True, which='both')
        # axes[x].set_xlabel(mode)
        axes[x].set_ylabel("Frequency")
        x_text = np.amax(error)
        axes[x].text(x_text, label_ypos, f'{mode}', horizontalalignment='center', fontsize=12)
        if pct is not None:
            # for pct in pcts:
            thresh = fixed_thresh(pct, error, 1)
            axes[x].plot([thresh[0], thresh[0]], [0, np.amax(y)], '--', label=f'{pct} * Max(Error)')
            axes[x].text(thresh[0], int(np.amax(y)/2), f'{sig_fig_round(thresh[0], 3)}', horizontalalignment='right', fontsize=12)
            thresholds.append(thresh[0])
        x += 1

    plt.subplots_adjust(wspace=0.15, hspace=0.15, top=0.95, right=0.98, bottom=0.08, left=0.05)

    fig.suptitle(f"Different Types of {traintest} Error Distribution - {model_name}")
    diag_file_name = f'hist-types-error-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center right')

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'hist-types-error-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
    
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return thresholds, plt

def fixed_thresh(pct, error, train_len):
    thresh_err = [pct * np.amax(error)] * train_len #or Define 90% value of max as threshold.
    thresh_err = np.array(thresh_err).flatten()
    return thresh_err

def anomaly_df(train, win_size, error, thresh):
    #Capture all details in a DataFrame for easy plotting
    df = pd.DataFrame(train[win_size:])
    df['error'] = np.max(error, axis=1)
    df['thresh'] = thresh
    df['anomaly'] = df['error'] > df['thresh']

    return df

def minutes(start_min, stop_min):
    '''Takes a start and end min and returns a datetime tuple for graph production'''

    tstart = datetime.datetime(2011, 12, 1, 11, start_min)
    tend = datetime.datetime(2011, 12, 1, 11, stop_min)

    return (tstart, tend)

def mid_time(dt_tup):
    a = dt_tup[0]
    b = dt_tup[1]
    mid = a + (b - a)/2

    return mid

def multiple_error_plot_and_save(dfs, modes, model_name, file_path, thresholds=None, save_new=True):
    all_errors = pd.DataFrame()
    for df, mode in zip(dfs, modes):
        all_errors[mode] = df['error']
    all_errors['Date'] = df['Date']

    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']

    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 8), sharex=True)
    x = 0
    if thresholds is None:
        thresholds = [0] * len(all_errors)
        
    for error_type, threshold in zip(modes, thresholds):
        all_errors['Threshold'] = [threshold] * len(all_errors)
        all_errors.plot(kind='line', x='Date', y=error_type, ax=axes[x], color=colours[x])
        if thresholds is not None:
            all_errors.plot(kind='line', x='Date', y='Threshold', ax=axes[x],color='red', linestyle='dashed')
        axes[x].set_ylabel(error_type)
        axes[x].grid(True, which='both')
        x += 1

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    fig.suptitle(f"Different Types of Test Reconstruction Error - {model_name}")
    diag_file_name = f'types-error-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'types-error-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
    
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return plt

def error_vs_thresh_plot_and_save(df, mode, model_name, file_path, ylimits=None, pct=None, tstart=None, tend=None, regions=True, save_new=True):
    #Plot testMAE vs max_trainMAE
    if tstart is not None and tend is not None:
        tstart = pd.to_datetime(tstart)
        tend = pd.to_datetime(tend)
        # df['Date'] = pd.to_datetime(df['Date'])
        plot_df = df.loc[(df['Date'] >= tstart) & (df['Date'] <= tend)]
        time_slice = True
        plt.figure(figsize=(8, 8))
        plt.xlim((tstart,tend))
    else:
        plot_df = df.copy(deep=True)
        time_slice = False
        plt.figure(figsize=(16, 8))

    if regions:
        if ylimits is None:
            y_min = min(plot_df['error'])
            y_max = max(plot_df['error'])
        else:
            y_min = ylimits[0]
            y_max = ylimits[1]

        legs_anom3 = minutes(5,6)
        plt.fill_between(legs_anom3, y_min, y_max, facecolor='yellow', alpha=0.2)
        if tstart is None or mid_time(legs_anom3) > tstart:
            plt.text(mid_time(legs_anom3), y_max-0.05, '1', horizontalalignment='center')

        full_anom6 = minutes(6, 8)
        plt.fill_between(full_anom6, y_min, y_max, facecolor='red', alpha=0.2)
        if tstart is None or mid_time(full_anom6) > tstart:
            plt.text(mid_time(full_anom6), y_max-0.05, '2', horizontalalignment='center')

        full_anom1 = minutes(9, 16)
        plt.fill_between(full_anom1, y_min, y_max, facecolor='orange', alpha=0.2)
        if tstart is None or mid_time(full_anom1) > tstart:
            plt.text(mid_time(full_anom1), y_max-0.05, '3', horizontalalignment='center')

        full_anom3 = minutes(17, 19)
        plt.fill_between(full_anom3, y_min, y_max, facecolor='red', alpha=0.2)
        if tstart is None or mid_time(full_anom3) > tstart:
            plt.text(mid_time(full_anom3), y_max-0.05, '4', horizontalalignment='center')

        legs_anom1 = minutes(20, 21)
        plt.fill_between(legs_anom1, y_min, y_max, facecolor='yellow', alpha=0.2)
        if tstart is None or mid_time(legs_anom1) > tstart:
            plt.text(mid_time(legs_anom1), y_max-0.05, '5', horizontalalignment='center')

        full_anom5 = minutes(23, 25)
        plt.fill_between(full_anom5, y_min, y_max, facecolor='red', alpha=0.2)
        if tstart is None or mid_time(full_anom5) > tstart:
            plt.text(mid_time(full_anom5), y_max-0.05, '6', horizontalalignment='center')

        full_anom4 = minutes(25, 26)
        plt.fill_between(full_anom4, y_min, y_max, facecolor='green', alpha=0.2)
        if tstart is None or mid_time(full_anom4) > tstart:
            plt.text(mid_time(full_anom4), y_max-0.05, '7', horizontalalignment='center')

        legs_anom2 = minutes(26, 27)
        plt.fill_between(legs_anom2, y_min, y_max, facecolor='yellow', alpha=0.2)
        if tstart is None or mid_time(legs_anom2) > tstart:
            plt.text(mid_time(legs_anom2), y_max-0.05, '8', horizontalalignment='center')

        full_anom2 = minutes(28, 32)
        plt.fill_between(full_anom2, y_min, y_max, facecolor='red', alpha=0.2)
        if tstart is None or mid_time(full_anom2) > tstart:
            plt.text(mid_time(full_anom2), y_max-0.05, '9', horizontalalignment='center')

        full_anom_fin = minutes(34, 36)
        plt.fill_between(full_anom_fin, y_min, y_max, facecolor='red', alpha=0.2)
        if tstart is None or mid_time(full_anom_fin) > tstart:
            plt.text(mid_time(full_anom_fin), y_max-0.05, '10', horizontalalignment='center')

    sns.lineplot(x=plot_df['Date'], y=plot_df['error'], label=mode)
    if pct is not None:
        sns.lineplot(x=plot_df['Date'], y=plot_df['thresh'], label='Threshold')
        title_end = f'- Fixed Threshold={pct}'
    else:
        title_end = ''

    if ylimits is not None:
        plt.ylim(ylimits)

    plt.xlabel("Date")
    if mode == 'MAE':
        plt.ylabel(mode)
        plt.title(f"MAE for Reconstructed Signal{title_end} - {model_name}")
    elif mode == 'Area':
        plt.ylabel(mode)
        plt.title(f"Area Difference for Reconstructed Signal{title_end} - {model_name}")
    elif mode == 'DTW':
        plt.ylabel(mode)
        plt.title(f"DTW Difference for Reconstructed Signal{title_end} - {model_name}")
    else:
        pass
    plt.legend()
    diag_file_name = f'error-thresh-{time_slice}-{pct}-{mode}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'error-thresh-{time_slice}-{pct}-{mode}-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
    
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    anomalies = df.loc[df['anomaly'] == True]

    return anomalies, plt

def detect_anom_plot_and_save(no_feats, sensors, df, anomalies, pct, mode, model_name, file_path, save_new=True):
    fig, axes = plt.subplots(no_feats, 1, figsize=(16, 16))
    for x in range(no_feats):
        df.plot(kind='line', x='Date', y=sensors[x], ax=axes[x])
        anomalies.plot(kind='scatter', x='Date', y=sensors[x], color='r', ax=axes[x], label='Anomaly')

    
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    fig.suptitle(f"Detected Anomalies - Fdixed Threshold={pct} - {mode} - {model_name}")
    plt.legend()
    diag_file_name = f'detected-anomalies-{pct}-{mode}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'detected-anomalies-{pct}-{mode}-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
    
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

def normal_dis_solver(x, xhat, mode='MAE'):
    if mode == 'MAE':
        diff = xhat - x
        error = np.mean(diff, axis=1)
    elif mode == 'Area':
        area_diff = []
        for x_win, xhat_win in zip(x, xhat):

            win_diff = []
            for i in range(x_win.shape[1]):

                x_int = integrate.trapz(x_win[:,i], dx=0.05)
                xhat_int = integrate.trapz(xhat_win[:,i], dx=0.05)
                diff = (xhat_int - x_int)/(2*x_win.shape[0])
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
    
    error_stats = []
    for sig_no in range(error.shape[1]):
        mu = np.mean(error[:,sig_no])
        sigma = np.std(error[:,sig_no])

        stats = {
            'mean': mu,
            'std': sigma
            }

        error_stats.append(stats)

    # anomaly scorer
    anomaly_score = np.array(error)
    for sig_no in range(error.shape[1]):

        # for point in range(error.shape[0]):
        e = np.abs(error[:,sig_no])
        mu = np.abs(error_stats[sig_no]['mean'])
        sigma = error_stats[sig_no]['mean']

        anomaly_score[:,sig_no] = (e - mu)/np.sum(e - mu)

    return error, error_stats, anomaly_score

def recombine_signals(array, mode='mean'):
    '''The inverse of the windowed signals'''

    if mode == 'mean':
        signals = np.empty((array.shape[0] + array.shape[1], array.shape[2]))
        i = 0
        for window in array:
            # if i % 1000 == 0:
                # print(f"{i}/{array.shape[0]}")
            curr_arr = signals[i:i+array.shape[1], :]
            signals[i:i+array.shape[1], :] = np.mean([window, curr_arr], axis=0)
            i += 1
        print(signals.shape)

    elif mode == 'median':
        signals_mat = np.empty((array.shape[0] + array.shape[1], array.shape[0] + array.shape[1], array.shape[2]))
        print(signals_mat.shape)
        i = 0
        for window in array:
            signals_mat[i, i:i+array.shape[1], :] = window
            i += 1

        signals = []
        for col in range(signals_mat.shape[1]):
            med = np.apply_along_axis(lambda v: np.median(v[v!=0]), 0, signals_mat[:, col])
            med[np.isnan(med)] = 0.
            signals.append(med)
            # if col % 1000 == 0:
                # print(med)
                # print(f"{col}/{signals_mat.shape[1]}")
        signals = np.array(signals)
        print(signals.shape)
    else:
        signals = None

    return signals

def full_reconstruct_plot(df, mode, name, file_path, tstart=None, tend=None, save_on=True):

    sensors = [['back_angle', 'back_hat'], ['left_angle', 'left_hat'], ['right_angle', 'right_hat']]
    labels = ['Orig.', 'Recon.']
    colours = [['#1f77b4', '#ff7f0e'], ['#2ca02c', '#d62728'], ['#9467bd', '#7f7f7f']]

    if tstart is not None and tend is not None:
        tstart = pd.to_datetime(tstart)
        tend = pd.to_datetime(tend)
        # df['Date'] = pd.to_datetime(df['Date'])
        plot_df = df.loc[(df['Date'] >= tstart) & (df['Date'] <= tend)]
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharey=True, sharex=True)
    else:
        plot_df = df.copy(deep=True)
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharey=True, sharex=True)

    
    fig.suptitle(f"Original vs. Reconstructed Signal - {name}")

    i = 0
    for sensor in sensors:

        plot_df.plot(kind='line', x='Date', y=sensor, ax=axes[i], label=labels, color=colours[i])
        i += 1

    axes[0].set_ylabel("Norm. Back Angle")
    axes[0].grid(True, which='both')

    axes[1].set_ylabel("Norm. Left Angle")
    axes[1].grid(True, which='both')

    axes[2].set_ylabel("Norm. Right Angle")
    axes[2].grid(True, which='both')

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.legend()

    diag_file_name = f'orig-recon-{mode}-{name}.png'
    full_path = f'{file_path}{diag_file_name}'

    if save_on:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'orig-recon-{mode}-{name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
 
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return plt

def moving_thresholds(error_dfs, win_size_mult=3, step_size_mult=30, std_thresh_mult=4):
    '''Function to calculate the moving threshold for each error inputted based on TadGAN method'''
    T = error_dfs[0].shape[0]
    # print(T)
    half_win_size = int(T/(win_size_mult*2))
    step_size = int(T/step_size_mult)
    anomalies_df = []
    for error_df in error_dfs:
        # error_df["Threshold"] = [0.0] * T
        for index in range(0, T-1, step_size):
            if (index + half_win_size < T) and (index - half_win_size > 0): 
                # print(f'{index - half_win_size}:{index + half_win_size}')
                window = error_df['error'].iloc[index - half_win_size : index + half_win_size]
                # print(window)
            elif (index + half_win_size > T):
                # print(f'{index - half_win_size}:{T}')
                window = error_df['error'].iloc[index - half_win_size : T]
                # print(window)
            elif (index - half_win_size < 0):
                # print(f'{0}:{index + half_win_size}')
                window = error_df['error'].iloc[0 : index + half_win_size]
                # print(window)
            else:
                print(f'UNKNOWN')

            win_mean = window.mean()
            win_std = window.std()
            win_anom_thresh = win_mean + (win_std * std_thresh_mult)
            # print(win_anom_thresh)

            # print(f'{index}:{index+step_size}')
            # print(index)
            error_df.loc[index:index+step_size,['Threshold']] = win_anom_thresh
            # print(index)
            # print(error_df.loc[index:index+step_size,['Threshold']])
        error_df['anomaly'] = (error_df['error'] > error_df['Threshold']) & (error_df['Threshold'] != 0)
        anomalies_df.append(error_df.loc[error_df['anomaly'] == True])
        # print(error_df)

    
    return error_dfs, anomalies_df

def multiple_error_moving_thresh_plot_and_save(error_dfs, modes, model_name, file_path, save_new=True):
    # all_errors = pd.DataFrame()
    # for df, mode in zip(dfs, modes):
    #     all_errors[mode] = df['error']
    # all_errors['Date'] = df['Date']

    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
    fig, axes = plt.subplots(len(error_dfs), 1, figsize=(16, 8), sharex=True)

    x = 0
    for error_df, error_type in zip(error_dfs, modes):
        error_df.plot(kind='line', x='Date', y='error', ax=axes[x], color=colours[x], label=error_type)
        error_df.plot(kind='line', x='Date', y='Threshold', ax=axes[x],color='red', linestyle='dashed')
        axes[x].set_ylabel(error_type)
        axes[x].grid(True, which='both')
        x += 1

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    fig.suptitle(f"Different Types of Test Reconstruction Error - {model_name}")
    diag_file_name = f'types-error-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'types-error-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
    
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return plt

def anomaly_pruning(anomaly_dfs, theta=0.1):
    '''At first, for each anomalous sequence, we use the maximum
    anomaly score to represent it, obtaining a set of maximum
    values {aimax, i = 1, 2, . . . , K}. Once these values are sorted
    in descending order, we can compute the decrease percent
    pi = (ai−1 max − ai max)/ai−1 max. When the first pi does not exceed
    a certain threshold θ (by default θ = 0.1), we reclassify all
    subsequent sequences (i.e., {aj seq, i ≤ j ≤ K}) as normal.'''

    for anomaly_df in anomaly_dfs:
        