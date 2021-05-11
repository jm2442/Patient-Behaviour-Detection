'''User defined modules'''

# Import required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import integrate
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math
import datetime
import os.path

def history_plot_and_save(history, model_name, file_path, ylimits=None, save_new=True):
    '''Generates and plot of the model's training history'''
    # Show the history of loss
    plt.figure(figsize=(5, 8))
    # Two plots
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    # Figure formatting
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Training - {model_name}")
    plt.legend()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')
    # Set limits on y axis if required
    if ylimits is not None:
        plt.ylim(ylimits)
    # Set file name for saving
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

def to_sequences(x, seq_size=1):
    '''Function to return windowed versions'''
    # Loop through the signal and extract window portions along it until they cannot be filled any more
    x_values = []
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
    return np.array(x_values)

def error_computation(x, xhat, mode='MAE'):
    '''Function to compute the error for each window based on selected metric'''
    if mode == 'MAE':
        # Find the difference, then average
        diff = np.abs(xhat - x)
        error = np.mean(diff, axis=1)
    elif mode == 'Area':
        # Loop through each window
        area_diff = []
        for x_win, xhat_win in zip(x, xhat):
            # Loop through each signal
            win_diff = []
            for i in range(x_win.shape[1]):
                # Calculate the input and reconstr. area using trapz
                x_int = integrate.trapz(x_win[:,i], dx=0.05)
                xhat_int = integrate.trapz(xhat_win[:,i], dx=0.05)
                # Find the difference in areas and append to list
                diff = np.abs(xhat_int - x_int)/(2*x_win.shape[0])
                win_diff.append(diff)
            area_diff.append(win_diff)
        error = np.array(area_diff)
    elif mode == 'DTW':
        # Loop through each window
        area_diff = []
        j = 0
        for x_win, xhat_win in zip(x, xhat):
            # Loop through each signal
            win_diff = []
            for i in range(x_win.shape[1]):
                # Determine the similarity using fastdtw and append to list
                x_diff, _ = fastdtw(x_win[:,i], xhat_win[:,i], dist=euclidean)
                win_diff.append(x_diff)
            area_diff.append(win_diff)
            # Logic to print progress because of the length of time taken
            j += 1
            if j % 1000 == 0:
                print(f"{j}/{x.shape[0]}")
        error = np.array(area_diff)
    else:
        error = None
    return error

def fixed_thresh(pct, error, train_len):
    '''Function to calculate the fixed threshold based on provided percentage'''
    thresh_err = [pct * np.amax(error)] * train_len #or Define 90% value of max as threshold.
    thresh_err = np.array(thresh_err).flatten()
    return thresh_err

def anomaly_df(train, win_size, error, thresh):
    '''Function that creates an anomaly dataframe based on comparing the errors larger than the threshold'''
    # Capture all details in a DataFrame for easy plotting
    df = pd.DataFrame(train[win_size:])
    # Find max for all signals for each point in time
    df['error'] = np.max(error, axis=1)
    df['thresh'] = thresh
    # Compare to give bool column
    df['anomaly'] = df['error'] > df['thresh']
    return df

def moving_thresholds(error_dfs, win_size_mult=3, step_size_mult=30, std_thresh_mult=4):
    '''Function to calculate the moving threshold for each error inputted based on TadGAN method '''
    # This function is designed to deal with several error computation methods so error dfs must be a list of each individual error df
    T = error_dfs[0].shape[0]
    half_win_size = int(T/(win_size_mult*2))
    step_size = int(T/step_size_mult)
    # Loop through each error df
    anomalies_df = []
    moving_error_dfs = []
    for error_df in error_dfs:
        # Loop through the whole length of the signal according to input parameters 
        moving_error_df = error_df.copy(deep=True)
        moving_error_df.reset_index(drop=True, inplace=True)
        for index in range(0, T-1, step_size):
            # Logic to correct window edges based on padding on the edges
            if (index + half_win_size < T) and (index - half_win_size > 0): 
                window = moving_error_df['error'].iloc[index - half_win_size : index + half_win_size]
            elif (index + half_win_size > T):
                window = moving_error_df['error'].iloc[index - half_win_size : T]
            elif (index - half_win_size < 0):
                window = moving_error_df['error'].iloc[0 : index + half_win_size]
            else:
                print('UNKNOWN')
            # Using window's statistics, calculate its specific threshold
            win_mean = window.mean()
            win_std = window.std()
            win_anom_thresh = win_mean + (win_std * std_thresh_mult)
            # Set threshold for the specific indexes as calculated threshold
            moving_error_df.loc[index:index+step_size,['Threshold']] = win_anom_thresh
        # Logic to correct theshold if it was set to 0
        moving_error_df['anomaly'] = (moving_error_df['error'] > moving_error_df['Threshold']) & (moving_error_df['Threshold'] != 0)
        # Append the anomalies_df to a list for dealing with multiple at one time
        anomalies_df.append(moving_error_df.loc[moving_error_df['anomaly'] == True])
        moving_error_dfs.append(moving_error_df)
    return moving_error_dfs, anomalies_df

def anomaly_pruning(error_dfs, theta=0.1):
    '''Function to prune anomalies according to 
    Please be advised - this function was implemented for demonstrational purposes and not a computational efficient manner.
    At first, for each anomalous sequence, we use the maximum
    anomaly score to represent it, obtaining a set of maximum
    values {aimax, i = 1, 2, . . . , K}. 
    Once these values are sorted in descending order, we can compute the decrease percent pi = (ai−1 max − ai max)/ai−1 max. 
    When the first pi does not exceed a certain threshold θ (by default θ = 0.1), we reclassify all subsequent sequences (i.e., {aj seq, i ≤ j ≤ K}) as normal.'''
    # Once again, assuming these different error types are being computed at once
    retained_anomaly_dfs = []
    for error_df in error_dfs:
        retained_anomaly_df = error_df.copy(deep=True)
        anomalies = error_df.loc[error_df['anomaly'] == True]
        old_index = 0
        groups = []
        # Grouping anomalies next to one another
        for index, _ in anomalies.iterrows():
            if index == old_index + 1:
                if old_index != 0:
                    group.append([index, anomalies['error'].loc[index]])
            else:
                if old_index != 0:
                    groups.append(group)
                g_list = [index, anomalies['error'].loc[index]]
                group = []
                group.append(g_list)
            old_index = index
        if not groups:
            groups.append(group)
        # Sort each group by its maximum
        max_anoms = []
        full_anoms = []
        for collective in groups:
            collective = np.array(collective)
            sorted_group = collective[collective[:, 1].argsort()][::-1]
            max_anoms.append(sorted_group[0])
            full_anoms.append(sorted_group)
        max_anoms = np.array(max_anoms)
        sorted_maxes = max_anoms[max_anoms[:, 1].argsort()][::-1]
        retained_anoms = []
        # Prune based off of theta
        ps = []
        prev_error = 0
        for anom in sorted_maxes:
            error = anom[1]
            p = np.abs((prev_error - error)/error)
            ps.append(p)
            retained_anoms.append(anom)
            if p < theta:
                break
            prev_error = error
        # Match non-pruned anomlies indexs and their original groups 
        final_anomalies = []
        for j in range(len(retained_anoms)):
            for i in range(len(full_anoms)):
                if retained_anoms[j][0] == full_anoms[i][0][0]:
                    final_anomalies.append(full_anoms[i])
                    j+=1
                    break
        # Bit of getting back into original format
        anom_indexes = []
        for anom_index in final_anomalies:
            anom_indexes.extend(anom_index[:,0])
        # Logic to change anomaly values that were set as true by adaptive window
        for index, _ in error_df.iterrows():
            if index in anom_indexes:
                retained_anomaly_df['anomaly'].loc[index] = True
            else:
                retained_anomaly_df['anomaly'].loc[index] = False
        retained_anomaly_dfs.append(retained_anomaly_df)
    return retained_anomaly_dfs

def error_hist_and_save(error, mode, model_name, file_path, save_on=True):
    '''Function to plot reconstrunction error histogram for the different signals'''
    # Plot
    plt.figure(figsize=(16, 8))
    plt.hist(error, bins=30)
    # Figure formatting
    plt.xlabel(mode)
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {mode} - {model_name}")
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')
    # Set file name for saving
    diag_file_name = f'error-hist-{mode}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'
    if save_on:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'error-hist-{mode}-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return plt

def example_plot(eg_win, no_feats, win_size, x, xhat, model_name, file_path, docplot=2, save_on=True):
    '''Function to plot a chosen example window original signal and reconstruction signal'''
    train_plot = x[eg_win]
    predict_plot = xhat[eg_win]
    x_plot = list(range(win_size))
    # Set plot size
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
    # Plot the x and xhat signals for each sensor
    for sensor in range(no_feats):
        sns.lineplot(x=x_plot, y=train_plot[:,sensor], color='b', ax=axes[sensor], label='x')
        sns.lineplot(x=x_plot, y=predict_plot[:,sensor], color='r', ax=axes[sensor], label='xhat')
    # Figure formatting
    axes[0].set_ylabel("Norm. Back Angle")
    axes[0].grid(True, which='both')
    axes[1].set_ylabel("Norm. Left Angle")
    axes[1].grid(True, which='both')
    axes[2].set_ylabel("Norm. Right Angle")
    axes[2].grid(True, which='both')
    plt.ylim((0,1))
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.legend()
    # Set file name for saving
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

def sig_fig_round(number, sig_fig):
    '''Function to round a number to a certain amount of significant figures'''
    rounded_number = round(number, sig_fig - int(math.floor(math.log10(abs(number)))) - 1)
    return rounded_number

def multiple_error_hist_plot_and_save(errors, modes, model_name, file_path, traintest='', pct=None, save_new=True):
    '''Function to plot multiple reconstruction error histograms'''
    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 8))
    labels = ['Back Error', 'Left Error', 'Right Error']
    thresholds = []
    # Plot multiple error histograms
    x = 0
    for error, mode in zip(errors, modes):
        y, _, _ = axes[x].hist(error, bins=30, label=labels)
        axes[x].grid(True, which='both')
        # axes[x].set_xlabel(mode)
        axes[x].set_ylabel("Frequency")
        x_text = np.amax(error)
        axes[x].text(x_text, np.amax(y)/2, f'{mode}', horizontalalignment='center', fontsize=12)
        if pct is not None:
            # for pct in pcts:
            thresh = fixed_thresh(pct, error, 1)
            axes[x].plot([thresh[0], thresh[0]], [0, np.amax(y)], '--', label=f'{pct} * Max(Error)')
            axes[x].text(thresh[0], int(np.amax(y)/2), f'{sig_fig_round(thresh[0], 3)}', horizontalalignment='right', fontsize=12)
            thresholds.append(thresh[0])
        x += 1
    # Figure formatting
    plt.subplots_adjust(wspace=0.15, hspace=0.15, top=0.95, right=0.98, bottom=0.08, left=0.05)
    fig.suptitle(f"Different Types of {traintest} Error Distribution - {model_name}")
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center right')
    # Set file name for saving
    diag_file_name = f'hist-types-error-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'
    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'hist-types-error-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return thresholds, plt

def minutes(start_min, stop_min):
    '''Takes a start and end min and returns a datetime tuple for graph production'''
    tstart = datetime.datetime(2011, 12, 1, 11, start_min)
    tend = datetime.datetime(2011, 12, 1, 11, stop_min)

    return (tstart, tend)

def mid_time(dt_tup):
    '''Function to find the mid point between two datetimes'''
    a = dt_tup[0]
    b = dt_tup[1]
    mid = a + (b - a)/2

    return mid

def multiple_error_plot_and_save(dfs, modes, model_name, file_path, thresholds=None, save_new=True):
    '''Function to plot multiple different error signals depending on calculation'''
    all_errors = pd.DataFrame()
    for df, mode in zip(dfs, modes):
        all_errors[mode] = df['error']
    all_errors['datetime'] = df['datetime']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 8), sharex=True)
    x = 0
    if thresholds is None:
        thresholds = [0] * len(all_errors)
    # Plot each error in different subplot
    for error_type, threshold in zip(modes, thresholds):
        all_errors['Threshold'] = [threshold] * len(all_errors)
        all_errors.plot(kind='line', x='datetime', y=error_type, ax=axes[x], color=colours[x])
        if thresholds != [0] * len(all_errors):
            all_errors.plot(kind='line', x='datetime', y='Threshold', ax=axes[x],color='red', linestyle='dashed')
        axes[x].set_ylabel(error_type)
        axes[x].grid(True, which='both')
        x += 1
    # Figure formatting
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')
    fig.suptitle(f"Different Types of Test Reconstruction Error - {model_name}")
    # Set file name for saving
    if thresholds == [0] * len(all_errors):
        threshold = None
    else:
        threshold = sig_fig_round(thresholds[0],2)
    diag_file_name = f'types-error-thresh{threshold}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'
    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'types-error-thresh{threshold}-{model_name}-{i}.png'
            full_path = f'{file_path}{diag_file_name}'
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Saved as: {full_path}")

    return plt

def error_vs_thresh_plot_and_save(df, mode, model_name, file_path, ylimits=None, pct=None, tstart=None, tend=None, regions=True, save_new=True):
    '''Function to plot the error reconstruction signal'''
    # If tstart and tend are defined then slice the plot
    if tstart is not None and tend is not None:
        tstart = pd.to_datetime(tstart)
        tend = pd.to_datetime(tend)
        plot_df = df.loc[(df['datetime'] >= tstart) & (df['datetime'] <= tend)]
        time_slice = True
        plt.figure(figsize=(8, 8))
        plt.xlim((tstart,tend))
    else:
        plot_df = df.copy(deep=True)
        time_slice = False
        plt.figure(figsize=(16, 8))
    # Plot regions of interest if set to true
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
    # Plot error signal
    sns.lineplot(x=plot_df['datetime'], y=plot_df['error'], label=mode)
    if pct is not None:
        # Plot threshold
        sns.lineplot(x=plot_df['datetime'], y=plot_df['thresh'], label='Threshold')
        title_end = f'- Fixed Threshold={pct}'
    else:
        title_end = ''
    # Limit y axis if it is set
    if ylimits is not None:
        plt.ylim(ylimits)
    # Plot axes labels
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
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')
    # Set file name for saving
    diag_file_name = f'error-thresh-{time_slice}-{pct}-{mode}-{model_name}.png'
    full_path = f'{file_path}{diag_file_name}'
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

def multiple_error_moving_thresh_plot_and_save(error_dfs, modes, model_name, file_path, pruned=False, save_new=True):
    '''Function to plot multiple error plots and the moving threshold'''
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
    fig, axes = plt.subplots(len(error_dfs), 1, figsize=(16, 8), sharex=True)
    # Looping plot of error and thresholds
    x = 0
    for error_df, error_type in zip(error_dfs, modes):
        error_df.plot(kind='line', x='datetime', y='error', ax=axes[x], color=colours[x], label=error_type)
        error_df.plot(kind='line', x='datetime', y='Threshold', ax=axes[x],color='red', linestyle='dashed')
        if pruned:
            error_df[error_df['anomaly'] == True].plot(kind='scatter', x='datetime', y='error', ax=axes[x],color='green', label='Preserved Anomalies')
        axes[x].set_ylabel(error_type)
        axes[x].grid(True, which='both')
        x += 1
    # Figure formatting
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')
    fig.suptitle(f"Different Types of Test Reconstruction Error - {model_name}")
    # Set file name for saving
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

def final_detection_plot_and_save(df, anomaly_dfs, methods, model_name,file_path, normalised=True, ROIs=False, save_new=True):
    '''Function to plot imposed detected anomalies on the normalised patient data'''
    # Plot patient data
    axes = df[['back_angle', 'left_angle', 'right_angle', 'datetime']].plot(subplots=True, kind='line', x='datetime', legend=True, sharey=True, figsize=(16, 8), zorder=0)
    # Use lists for looping plots
    angles = ['back_angle', 'left_angle', 'right_angle']
    colours = ['#d62728', '#e377c2', '#bcbd22', '#17becf']
    if normalised:
        y_heights = [.05, .1, .15, .2]
    else:
        y_heights = [2.5, 5, 7.5, 10]
    markers = ['v', 'v', 'v', 'v']
    # Plot detected anomalies using markers
    x = 0
    for anomaly in anomaly_dfs:
        tstart = anomaly.iloc[0]['datetime']
        anomalies = anomaly.loc[anomaly['anomaly'] == True]
        if anomalies.empty:
            x += 1
            continue
        j = 0
        for angle in angles:
            max_angle = max(df.loc[:,angle])
            anomalies.loc[:,[f'{angle}_y_heights']] = [(y_heights[x] + max_angle)] * anomalies.shape[0]
            anomalies.plot(kind='scatter', x='datetime', y=f'{angle}_y_heights', ax=axes[j], color=colours[x], label=methods[x], marker=markers[x])
            j += 1
        x += 1
    # Plot regions of interest if set to true
    if ROIs and normalised:
        legs_anom3 = minutes(5,6)
        axes[1].fill_between(legs_anom3, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(legs_anom3, 0, 1, facecolor='yellow', alpha=0.2)
        if mid_time(legs_anom3) > tstart:
            axes[1].text(mid_time(legs_anom3), 0.1, '1', horizontalalignment='center')

        full_anom6 = minutes(6, 8)
        axes[0].fill_between(full_anom6, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom6, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom6, 0, 1, facecolor='red', alpha=0.2)
        
        if mid_time(full_anom6) > tstart:
            axes[1].text(mid_time(full_anom6), 0.1, '2', horizontalalignment='center')

        full_anom1 = minutes(9, 16)
        axes[0].fill_between(full_anom1, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(full_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        
        if mid_time(full_anom1) > tstart:
            axes[1].text(mid_time(full_anom1), 0.1, '3', horizontalalignment='center')

        full_anom3 = minutes(17, 19)
        axes[0].fill_between(full_anom3, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom3, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom3, 0, 1, facecolor='red', alpha=0.2)
        
        if mid_time(full_anom3) > tstart:
            axes[1].text(mid_time(full_anom3), 0.1, '4', horizontalalignment='center')

        legs_anom1 = minutes(20, 21)
        axes[1].fill_between(legs_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(legs_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        
        if mid_time(legs_anom1) > tstart:
            axes[1].text(mid_time(legs_anom1), 0.1, '5', horizontalalignment='center')

        full_anom5 = minutes(23, 25)
        axes[0].fill_between(full_anom5, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom5, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom5, 0, 1, facecolor='red', alpha=0.2)
        
        if mid_time(full_anom5) > tstart:
            axes[1].text(mid_time(full_anom5), 0.1, '6', horizontalalignment='center')

        full_anom4 = minutes(25, 26)
        axes[0].fill_between(full_anom4, 0, 1, facecolor='green', alpha=0.2)
        axes[1].fill_between(full_anom4, 0, 1, facecolor='green', alpha=0.2)
        axes[2].fill_between(full_anom4, 0, 1, facecolor='green', alpha=0.2)
        
        if mid_time(full_anom4) > tstart:
            axes[1].text(mid_time(full_anom4), 0.1, '7', horizontalalignment='center')

        legs_anom2 = minutes(26, 27)
        axes[1].fill_between(legs_anom2, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(legs_anom2, 0, 1, facecolor='yellow', alpha=0.2)
        
        if mid_time(legs_anom2) > tstart:
            axes[1].text(mid_time(legs_anom2), 0.1, '8', horizontalalignment='center')

        full_anom2 = minutes(28, 32)
        axes[0].fill_between(full_anom2, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom2, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom2, 0, 1, facecolor='red', alpha=0.2)
        
        if mid_time(full_anom2) > tstart:
            axes[1].text(mid_time(full_anom2), 0.1, '9', horizontalalignment='center')


        full_anom_fin = minutes(34, 36)
        axes[0].fill_between(full_anom_fin, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom_fin, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom_fin, 0, 1, facecolor='red', alpha=0.2)
        
        if mid_time(full_anom_fin) > tstart:
            axes[1].text(mid_time(full_anom_fin), 0.1, '10', horizontalalignment='center')
    # Set y label axes
    if normalised:
        axes[0].set_ylabel("Norm. Back Angle")
        axes[1].set_ylabel("Norm. Left Angle")
        axes[2].set_ylabel("Norm. Right Angle")
        text = 'Normalised '
    else:
        axes[0].set_ylabel("Backrest Angle (Deg)")
        axes[1].set_ylabel("Left Hip Angle (Deg)")
        axes[2].set_ylabel("Right Hip Angle (Deg)")
        text = ''
    # Set axes formatting
    axes[0].set_title(f"Detected Anomalies within {text}Patient Data - {model_name}")
    axes[0].grid(True, which='both')
    axes[1].grid(True, which='both')
    axes[2].set_xlabel("01/12/2011 - Time")
    axes[2].grid(True, which='both')
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.legend(loc = 'lower right')
    # Set file name for saving
    filename = f'{text.lower()}detected-anomalies-output-{model_name}.png'
    full_path = f'{file_path}{filename}'
    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            filename = f'{text.lower()}detected-anomalies-output-{model_name}-{i}.png'
            full_path = f'{file_path}{filename}'

        print(f"Saved as: {full_path}")
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
    
    return plt
