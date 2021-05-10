'''A script to retrieve and prep the patient data into a form ready for preprocessing steps'''

# Import required libraries
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from control_data import compute_angle_derivs
from sklearn.preprocessing import MinMaxScaler

def data_extractor(file_path, file_name):
    '''Extract csv into df'''

    df = pd.read_csv(f'{file_path}{file_name}', index_col=False, skiprows=[0,1,2,4])

    return df

def data_filterer(df):
    '''Only extract the angles from the measurement sensors and rename the columns'''

    keep_cols = ["Backrest Angle / Deg", "Left Hip Angle / Deg", "Right Hip Angle / Deg", "Date"]

    angles_df = df[keep_cols]

    angles_df = angles_df.rename(columns={
        "Backrest Angle / Deg": "back_angle", 
        "Left Hip Angle / Deg": "left_angle", 
        "Right Hip Angle / Deg": "right_angle"
        })

    angles_df.loc[:,'Date'] = pd.to_datetime(df.Date.astype(str)+' '+df.Time.astype(str))

    return angles_df

def datetime_converter(df, freq=20):
    '''Convert column containing dates into datetime with appropriate frequency interval'''

    date = df.loc[:,'Date'][0]
    dtime = [date]
    for _ in range(df.shape[0]-1):
        date += datetime.timedelta(milliseconds=int((1/freq)*1e3))
        dtime.append(date)

    new_df = df.copy(deep=True)
    new_df.loc[:,'datetime'] = dtime
    new_df.drop(['Date'], axis=1 , inplace=True)

    return new_df

def plot_patient_orig_data(df, diagram_folder, save_new=True):
    '''Plot angle data provided originally'''

    axes = df[['back_angle', 'left_angle', 'right_angle', 'datetime']].plot(subplots=True, kind='line', x='datetime', legend=False, sharey=True, figsize=(16, 8))

    axes[0].set_title("Provided Patient Chair Data")
    axes[0].set_ylabel("Backrest Angle (Deg)")
    axes[0].grid(True, which='both')

    axes[1].set_ylabel("Left Hip Angle (Deg)")
    axes[1].grid(True, which='both')

    axes[2].set_ylabel("Right Hip Angle (Deg)")
    axes[2].set_xlabel("01/12/2011 - Time")
    axes[2].grid(True, which='both')

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    filename = 'patient-original.png'
    full_path = f'{diagram_folder}{filename}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'patient-original-{i}.png'
            full_path = f'{diagram_folder}{diag_file_name}'


        plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.show()

def backlock_remover(df, start_index, end_index, backlock_on):
    '''Remove period of backrest lock to prevent bias'''

    new_df = df.copy(deep=True)
    if backlock_on ==False:
        drop_range = list(range(start_index, end_index+1))
        new_df.drop(new_df.index[drop_range], inplace=True)
        new_df.reset_index(inplace=True)

    return new_df

def plot_backlock_removed_data(df, diagram_folder, save_new=True):
    '''Plot angle data with backrest lock period removed'''

    axes = df[['back_angle', 'left_angle', 'right_angle', 'datetime']].plot(subplots=True, kind='line', x='datetime', legend=False, sharey=True, figsize=(16, 8))

    axes[0].set_title("Patient Data with Period of Backlock Removed")
    axes[0].set_ylabel("Backrest Angle (Deg)")
    axes[0].grid(True, which='both')

    axes[1].set_ylabel("Left Hip Angle (Deg)")
    axes[1].grid(True, which='both')

    axes[2].set_ylabel("Right Hip Angle (Deg)")
    axes[2].set_xlabel("01/12/2011 - Time")
    axes[2].grid(True, which='both')

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)
    plt.grid(True, which='both')

    filename = 'patient-backout.png'
    full_path = f'{diagram_folder}{filename}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'patient-backout-{i}.png'
            full_path = f'{diagram_folder}{diag_file_name}'
        plt.savefig(full_path, bbox_inches='tight', dpi=300)

    plt.show()

def patient_normaliser(df):
    '''Normalise the angles computed for each subject's action'''

    angles = ['back_angle', 'left_angle', 'right_angle']
    norm_df = df.copy(deep=True)
    for angle in angles:

        scaler = MinMaxScaler()
        temp_angle = scaler.fit_transform(np.array(norm_df[angle]).reshape(-1, 1))
        norm_df[angle] = temp_angle

    return norm_df

def minutes(start_min, stop_min):
    '''Takes a start and end min and returns a datetime tuple for graph production'''

    tstart = datetime.datetime(2011, 12, 1, 11, start_min)
    tend = datetime.datetime(2011, 12, 1, 11, stop_min)

    return (tstart, tend)

def mid_time(dt_tup):
    '''Find the mid point of two datetimes'''

    a = dt_tup[0]
    b = dt_tup[1]
    mid = a + (b - a)/2

    return mid

def plot_patient_norm_data(df, anom_on, diagram_folder, save_new=True):
    '''Plots the normalised patient dataset with option to include author's attempt at suggested regions'''

    axes = df[['back_angle', 'left_angle', 'right_angle', 'datetime']].plot(subplots=True, kind='line', x='datetime', legend=False, sharey=True, figsize=(16, 8))

    if anom_on:
        axes[0].set_title("Normalised Patient Data - Annotated Regions of Interest")

        legs_anom3 = minutes(5,6)
        axes[1].fill_between(legs_anom3, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(legs_anom3, 0, 1, facecolor='yellow', alpha=0.2)
        axes[1].text(mid_time(legs_anom3), 0.1, '1', horizontalalignment='center')

        full_anom6 = minutes(6, 8)
        axes[0].fill_between(full_anom6, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom6, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom6, 0, 1, facecolor='red', alpha=0.2)
        axes[1].text(mid_time(full_anom6), 0.1, '2', horizontalalignment='center')

        full_anom1 = minutes(9, 16)
        axes[0].fill_between(full_anom1, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(full_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        axes[1].text(mid_time(full_anom1), 0.1, '3', horizontalalignment='center')

        full_anom3 = minutes(17, 19)
        axes[0].fill_between(full_anom3, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom3, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom3, 0, 1, facecolor='red', alpha=0.2)
        axes[1].text(mid_time(full_anom3), 0.1, '4', horizontalalignment='center')

        legs_anom1 = minutes(20, 21)
        axes[1].fill_between(legs_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(legs_anom1, 0, 1, facecolor='yellow', alpha=0.2)
        axes[1].text(mid_time(legs_anom1), 0.1, '5', horizontalalignment='center')

        full_anom5 = minutes(23, 25)
        axes[0].fill_between(full_anom5, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom5, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom5, 0, 1, facecolor='red', alpha=0.2)
        axes[1].text(mid_time(full_anom5), 0.1, '6', horizontalalignment='center')

        full_anom4 = minutes(25, 26)
        axes[0].fill_between(full_anom4, 0, 1, facecolor='green', alpha=0.2)
        axes[1].fill_between(full_anom4, 0, 1, facecolor='green', alpha=0.2)
        axes[2].fill_between(full_anom4, 0, 1, facecolor='green', alpha=0.2)
        axes[1].text(mid_time(full_anom4), 0.1, '7', horizontalalignment='center')

        legs_anom2 = minutes(26, 27)
        axes[1].fill_between(legs_anom2, 0, 1, facecolor='yellow', alpha=0.2)
        axes[2].fill_between(legs_anom2, 0, 1, facecolor='yellow', alpha=0.2)
        axes[1].text(mid_time(legs_anom2), 0.1, '8', horizontalalignment='center')

        full_anom2 = minutes(28, 32)
        axes[0].fill_between(full_anom2, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom2, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom2, 0, 1, facecolor='red', alpha=0.2)
        axes[1].text(mid_time(full_anom2), 0.1, '9', horizontalalignment='center')


        full_anom_fin = minutes(34, 36)
        axes[0].fill_between(full_anom_fin, 0, 1, facecolor='red', alpha=0.2)
        axes[1].fill_between(full_anom_fin, 0, 1, facecolor='red', alpha=0.2)
        axes[2].fill_between(full_anom_fin, 0, 1, facecolor='red', alpha=0.2)
        axes[1].text(mid_time(full_anom_fin), 0.1, '10', horizontalalignment='center')
    else:
        axes[0].set_title("Normalised Patient Data")

    axes[0].set_ylabel("Norm. Back Angle")
    axes[0].grid(True, which='both')

    axes[1].set_ylabel("Norm. Left Angle")
    axes[1].grid(True, which='both')

    axes[2].set_ylabel("Norm. Right Angle")
    axes[2].grid(True, which='both')

    axes[2].set_xlabel("01/12/2011 - Time")
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.98, bottom=0.08, left=0.05)

    filename = f'patient-norm-anom_{anom_on}.png'
    full_path = f'{diagram_folder}{filename}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'patient-norm-anom_{anom_on}-{i}.png'
            full_path = f'{diagram_folder}{diag_file_name}'

        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        
    plt.show()

def save_proc_patient_df(df, dest_file_path, backlock_on=False):
    '''Save preprocessed patient dataset into csv'''

    if backlock_on == False:
        new_file_name = 'patient_data.csv'
    else:
        new_file_name = 'patient_data_wlock.csv'

    if os.path.exists(f'{dest_file_path}{new_file_name}'):
        print(f"{new_file_name} created already, delete file if new version required")
    else:
        df.to_csv(f'{dest_file_path}{new_file_name}',index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
        print(f"Saved as: {dest_file_path}{new_file_name}")

def patient_angle_derivs(angle_df):
    '''Calculate the deriv of the angles computed'''

    angle_derivs_df = pd.DataFrame()

    for deriv in [1,2]:

        window_size = 5
        polyorder = 2
        angle_df = compute_angle_derivs(angle_df, window_size, polyorder, deriv)

    angle_derivs_df = angle_derivs_df.append(angle_df)

    return angle_derivs_df

def patient_display_derivs(angle_derivs_df):
    '''Script to plot the deriv of the angles computed'''
    _, axs = plt.subplots(3, 3)

    axs[0, 0].plot(angle_derivs_df['back_angle'])
    axs[0, 0].set_ylabel('Angle (deg)')
    axs[0, 0].set_title('Back')
    axs[0, 1].plot(angle_derivs_df['left_angle'])
    axs[0, 2].plot(angle_derivs_df['right_angle'])

    deriv = 1

    axs[1, 0].plot(angle_derivs_df[f'back_{deriv}der'])
    axs[1, 1].plot(angle_derivs_df[f'left_{deriv}der'])
    axs[1, 0].set_ylabel('1st Deriv Angle (deg/s)')
    axs[0, 1].set_title('Left')
    axs[1, 2].plot(angle_derivs_df[f'right_{deriv}der'])

    deriv = 2

    axs[2, 0].plot(angle_derivs_df[f'back_{deriv}der'])
    axs[2, 1].plot(angle_derivs_df[f'left_{deriv}der'])
    axs[2, 2].plot(angle_derivs_df[f'right_{deriv}der'])
    axs[2, 0].set_ylabel('2nd Deriv Angle (deg/s^2)')
    axs[0, 2].set_title('Right')

    plt.show()

def format_to_control(angle_derivs_df, d_file_path, save_on, backlock_on):
    '''Script to add columns to make csv similar to control data path'''

    extra_cols = ["action", "subject", "frame"]

    extra_cols_df = pd.DataFrame(columns=extra_cols)

    angle_derivs_df = extra_cols_df.append(angle_derivs_df, ignore_index=True)

    print(angle_derivs_df)

    if save_on:
        if backlock_on == False:
            new_file_name = 'patient_data.csv'
        else:
            new_file_name = 'patient_data_wlock.csv'
        if os.path.exists(f'{d_file_path}{new_file_name}'):
            print(f"{new_file_name} csv created already, delete file if new version required")
        else:
            angle_derivs_df.to_csv(f'{d_file_path}{new_file_name}',index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
            print(f'Saved a copy as csv in :{d_file_path}')

    return angle_derivs_df

def main():
    '''Run the script'''

    # Set constants and settings
    FILE_DIR = '../patient-simulator-FYP/'
    DIAGRAM_DIR = FILE_DIR + 'diagrams/'
    DATA_DIR = FILE_DIR + 'datasets/'
    PATIENT_DIR = DATA_DIR + 'patient/'
    plot_on = True
    save_new = False
    PATIENT_DATA_FILENAME = 'DE250053 - (Callibrated - new method - compressed).csv'

    # Extract data from originally provided csv
    patient_original_df = data_extractor(PATIENT_DIR, PATIENT_DATA_FILENAME)

    # Filter only relevant angle data
    patient_angles_df = data_filterer(patient_original_df)

    # Convert datetime column to proper datetime
    patient_orig_angles_df = datetime_converter(patient_angles_df)

    if plot_on:
        # plot originally provided angle data
        plot_patient_orig_data(patient_orig_angles_df, DIAGRAM_DIR, save_new=save_new)

    # Set parameters for what to remove from when removing backrest lock period
    start_index = 25000
    end_index = 51000
    keep_backlock = False

    # Remove backrest lock period
    patient_no_lock_df = backlock_remover(patient_angles_df, start_index, end_index, keep_backlock)

    # Convert datetime column to proper datetime
    patient_no_lock_df = datetime_converter(patient_no_lock_df)

    if plot_on:
        # plot angle data with period of backrest lock removed
        plot_backlock_removed_data(patient_no_lock_df, DIAGRAM_DIR, save_new=save_new)

    # Normalise patient dataset
    patient_norm_df = patient_normaliser(patient_no_lock_df)

    if plot_on:
        # Plot normalised data with option to include suggest periods of anomaly
        suggested_anom = True
        plot_patient_norm_data(patient_norm_df, suggested_anom, DIAGRAM_DIR, save_new=save_new)

    # Save a copy of preprocessed data as .csv
    save_proc_patient_df(patient_norm_df, PATIENT_DIR, keep_backlock)

    ### NO LONGER INCLUDED IN PIPELINE ###
    # patient_angle_derivs_df = angle_derivs(patient_norm_df)
    # if plot_on:
    #     patient_display_derivs(patient_angle_derivs_df)
    # format_to_control(patient_angle_derivs_df, PATIENT_DIR, False, keep_backlock)

if __name__ == "__main__":
    main()
