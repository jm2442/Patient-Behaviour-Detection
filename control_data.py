'''A script to retrieve and prep the control data into a form ready for preprocessing steps'''

# Import required libraries
from ast import literal_eval
import os.path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

def number_to_string(num_list):
    '''Basic function to aid with filename looping,
    converting single digit ints to 0 leading str versions'''

    str_list = []
    for num in num_list:
        if num < 10:
            str_list.append('0'+str(num))
        else:
            str_list.append(str(num))

    return str_list

def txt_extract_and_filter(file_path):
    '''Extracts and filters data from .txts'''

     # number of actions and subjects from the dataset
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))

    # list to append dicts to form df
    samples = []

    # loop through actions and subjects
    for action in number_to_string(action_nos):

        for subject in number_to_string(subject_nos):

            # concat filename for extraction
            file_name = f'sitting-skeleton-txts/a{action}_s{subject}_e01_skeleton.txt'

            # Open file as list of lines
            file_object = open(f'{file_path}{file_name}', 'r').readlines()

            frame_per_rows = []
            frame = []
            frame_count = -1

            # loop through lines and strip any trailing or
            # leading whites spaces before splitting into list of lists
            for line in file_object:

                stripped_line = line.strip()
                line_list = stripped_line.split()

                # Each new frame is noted by a single line 40 so this is used as the identifier
                if line_list[0] == '40':

                    # Prevent the initial two useless lines
                    # from each file causing an issue
                    if frame_count > -1:
                        frame_per_rows.append(frame)
                        frame = []

                    frame_count += 1

                elif frame_count > -1:

                    # Issue with one file found a13-s06,
                    #  so this is to notify that this one frame
                    # was disregarded.
                    try:

                        if line_list[3] == '1':

                            line_list = list(map(float, line_list))
                            frame.append(line_list[0:3])

                    except Exception as ex:
                        print(ex)
                        print(frame_count)
                        continue

            frame_no = 1
            # Loop through created list and convert info and
            # extracted into dict to allow easy conversion to csv
            for sample in frame_per_rows:

                df_dict = {
                    'action': action,
                    'subject': subject,
                    'frame_no': frame_no,
                    'neck': sample[2],
                    'bt_spine': sample[1],
                    'tp_pelvis': sample[0],
                    'lf_pelvis': sample[12],
                    'lf_knee': sample[13],
                    'rh_pelvis': sample[16],
                    'rh_knee': sample[17],
                }

                samples.append(df_dict)

                frame_no += 1

    samples_df = pd.DataFrame(samples)

    new_file_name = 'coords_data.csv'
    if os.path.exists(f'{file_path}{new_file_name}'):
        print(f"{new_file_name} created already, delete file if new version required")
    else:
        samples_df.to_csv(f'{file_path}{new_file_name}',index=False)
        print(f'Saved a copy as csv in :{file_path}{new_file_name}')

    return samples_df

def vectors_to_angle(vec1, vec2):
    '''Takes two vectors and calculates the angle between them'''

    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)

    return (angle/math.pi) * 180

def coords_to_angles(file_path):
    '''Converts 3D positions to angles'''

    file_name = 'coords_data.csv'
    df = pd.read_csv(f'{file_path}{file_name}', index_col=False)

    back_angle = []
    left_angle = []
    right_angle = []

    # loop through each row
    for _, row in df.iterrows():

        # Tricksy conversion from string to array, pds stores lists as strs
        neck_arr = np.array(literal_eval(row['neck']))
        bt_spine_arr = np.array(literal_eval(row['bt_spine']))
        tp_pelvis_arr = np.array(literal_eval(row['tp_pelvis']))
        lf_pelvis_arr = np.array(literal_eval(row['lf_pelvis']))
        lf_knee_arr = np.array(literal_eval(row['lf_knee']))
        rh_pelvis_arr = np.array(literal_eval(row['rh_pelvis']))
        rh_knee_arr = np.array(literal_eval(row['rh_knee']))

        # bt_spine -> neck
        bt_spine_neck = np.subtract(neck_arr, bt_spine_arr)
        # bt_spine -> tp_pelvis
        bt_spine_tp_pelvis = np.subtract(tp_pelvis_arr, bt_spine_arr)
        # lf_pelvis -> tp_pelvis
        lf_pelvis_tp_pelvis = np.subtract(tp_pelvis_arr, lf_pelvis_arr)
        # lf_pelvis -> lf_knee
        lf_pelvis_lf_knee = np.subtract(lf_knee_arr, lf_pelvis_arr)
        # rh_pelvis -> tp_pelvis
        rh_pelvis_tp_pelvis = np.subtract(tp_pelvis_arr, rh_pelvis_arr)
        # rh_pelvis -> rh_knee
        rh_pelvis_rh_knee = np.subtract(rh_knee_arr, rh_pelvis_arr)

        # Compute the angle between the vectors
        back_angle.append(vectors_to_angle(bt_spine_neck, bt_spine_tp_pelvis))
        left_angle.append(vectors_to_angle(lf_pelvis_tp_pelvis, lf_pelvis_lf_knee))
        right_angle.append(vectors_to_angle(rh_pelvis_tp_pelvis, rh_pelvis_rh_knee))

    columns_add = ['back_angle', 'left_angle', 'right_angle']
    
    angle_df = df.copy(deep=True)
    # Add columns to df
    angle_df[columns_add[0]] = back_angle
    angle_df[columns_add[1]] = left_angle
    angle_df[columns_add[2]] = right_angle
    
    angle_df.drop(labels=['neck','bt_spine','tp_pelvis','lf_pelvis','lf_knee','rh_pelvis','rh_knee'], inplace=True, axis=1)

    # Transform the values from the obtuse to acute
    for col in columns_add:
        angle_df[col] = 180 - angle_df[col]

    return angle_df

def resampler(df, old_hz=30, new_hz=20):
    '''Resamples data from 30Hz to 20Hz'''

    # Loop through each action and subject
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    angles = ['back_angle', 'left_angle', 'right_angle']

    resamp_headers = ['action', 'subject', 'frame',
    'back_angle', 'left_angle', 'right_angle']

    resamp_df = pd.DataFrame(columns=resamp_headers)

    for action in action_nos:
        
        for subject in subject_nos:

            angle_df = df[angles].loc[
                (df['action'] == action) & (df['subject'] == subject)]

            for angle in angles:

                # Extract data
                data = np.array(angle_df[angle].tolist())

                # Resample data
                number_of_samples = round(len(data) * float(new_hz) / old_hz)
                new_data = sps.resample(data, number_of_samples)

                # Smooth data
                window_size = 3
                smoothed_data = sps.savgol_filter(new_data, window_size, 2)

                if angle == 'back_angle':
                    back_angle_resamp = smoothed_data
                elif angle == 'left_angle':
                    left_angle_resamp = smoothed_data
                elif angle == 'right_angle':
                    right_angle_resamp = smoothed_data

            append_df = pd.DataFrame(columns=resamp_headers)
            num_frame = len(back_angle_resamp)
            append_df['action'] = [action] * num_frame
            append_df['subject'] = [subject] * num_frame
            append_df['frame'] = list(range(1, num_frame+1))
            append_df['back_angle'] = back_angle_resamp
            append_df['left_angle'] = left_angle_resamp
            append_df['right_angle'] = right_angle_resamp

            resamp_df = resamp_df.append(append_df, ignore_index=True)

    return resamp_df

def plot_control_actions(df, diagram_folder, save_new=True):
    '''Plots each of the control actions angles for all of the subjects'''

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    action_type = ['Drinking', 'Eating', 'Reading', 'Phoning', 'Writing', 'Laptop', 'Cheering', 'Nothing', 'Throwing a ball', 'Playing a video game', 'Playing a guitar']

    for action, label in zip(action_nos, action_type):

        fig, axs = plt.subplots(3, 1, sharey=True, figsize=(8, 8))
        fig.suptitle("MSR DailyActivity3D Processed Angles - Control Data")

        for subject in subject_nos:

            new_df = df.loc[
                (df['action'] == action) & (df['subject'] == subject)]
        
            axs[0].set_title(f"Control Data - Action: {label}")

            axs[0].plot(new_df['back_angle'])
            axs[0].set_ylabel('Back Angle (Deg)')

            axs[1].plot(new_df['left_angle'])
            axs[1].set_ylabel('Left Angle (Deg)')
            axs[2].plot(new_df['right_angle'], label=f'Subject {subject}')
            
            axs[2].set_ylabel('Right Angle (Deg)')
            axs[2].set_xlabel("Index")

            axs[0].grid(True, which='both')
            axs[1].grid(True, which='both')
            axs[2].grid(True, which='both')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.90, right=0.98, bottom=0.08, left=0.05)
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'center right')


        filename = f'control-{label}.png'
        full_path = f'{diagram_folder}{filename}'

        if save_new:
            i = 0
            while os.path.isfile(full_path):
                i += 1
                diag_file_name = f'control-{label}-{i}.png'
                full_path = f'{diagram_folder}{diag_file_name}'

            plt.savefig(full_path, bbox_inches='tight', dpi=300)

    plt.show()

def control_normaliser(df):
    '''Individually normalise the angles computed for each subject's action'''

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    angles = ['back_angle', 'left_angle', 'right_angle']

    normal_angles_df = pd.DataFrame()

    for action in action_nos:

        for subject in subject_nos:

            angles_df = df.loc[
                (df['action'] == action) & (df['subject'] == subject)]

            for angle in angles:

                scaler = MinMaxScaler()
                temp_angle = scaler.fit_transform(np.array(angles_df.loc[:, (angle)]).reshape(-1, 1))
                angles_df[angle] = temp_angle

            normal_angles_df = normal_angles_df.append(angles_df)

    return normal_angles_df

def plot_control_norm_data(df, diagram_folder, save_new=True):
    '''Plots the entire normalised control dataset on the same axes'''

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    action_type = ['Drinking', 'Eating', 'Reading', 'Phoning', 'Writing', 'Laptop', 'Cheering', 'Nothing', 'Throwing a ball', 'Playing a video game', 'Playing a guitar']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, axs = plt.subplots(3, 1, sharey=True, figsize=(16, 8))
    i = 0
    for action, label in zip(action_nos, action_type):
        
        for subject in subject_nos:

            new_df = df.loc[
                (df['action'] == action) & (df['subject'] == subject)]

        
            axs[0].set_title("Normalised Control Data")

            axs[0].plot(new_df['back_angle'])
            axs[0].set_ylabel('Norm. Back Angle')

            axs[1].plot(new_df['left_angle'])
            axs[1].set_ylabel('Norm. Left Angle')

            if i >= 100:
                axs[2].plot(new_df['right_angle'], color=colours[subject-1], label=f'Subject {subject}')
            else:
                axs[2].plot(new_df['right_angle'], color=colours[subject-1])
            
            axs[2].set_ylabel('Norm. Right Angle')
            axs[2].set_xlabel("Index")

            i += 1
            axs[0].grid(True, which='both')
            axs[1].grid(True, which='both')
            axs[2].grid(True, which='both')

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.90, right=0.98, bottom=0.08, left=0.05)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center right')


    filename = 'control-norm.png'
    full_path = f'{diagram_folder}{filename}'

    if save_new:
        i = 0
        while os.path.isfile(full_path):
            i += 1
            diag_file_name = f'control-norm-{i}.png'
            full_path = f'{diagram_folder}{diag_file_name}'
        plt.savefig(full_path, bbox_inches='tight', dpi=300)

    plt.show()

def save_proc_control_df(df, file_path):
    '''Save the preprocess data as a csv'''

    new_file_name = 'control_data.csv'

    if os.path.exists(f'{file_path}{new_file_name}'):
        print(f"{new_file_name} created already, delete file if new version required")
    else:
        df.to_csv(f'{file_path}{new_file_name}',index=False)
        print(f"Saved as: {file_path}{new_file_name}")

def compute_angle_derivs(angle_df, window_size, polyorder, deriv):
    '''Compute the angle derivs and return expanded df'''

    new_angle_df = angle_df.copy()

    new_angle_df[f'back_{deriv}der'] = sps.savgol_filter(
        angle_df['back_angle'],
        window_length=window_size,
        polyorder=polyorder,
        deriv=deriv
        )

    new_angle_df[f'left_{deriv}der'] = sps.savgol_filter(
        angle_df['left_angle'],
        window_length=window_size,
        polyorder=polyorder,
        deriv=deriv
        )

    new_angle_df[f'right_{deriv}der'] = sps.savgol_filter(
        angle_df['right_angle'],
        window_length=window_size,
        polyorder=polyorder,
        deriv=deriv
        )

    return new_angle_df

def control_angle_derivatives(df):
    '''Calculate the deriv of the angles computed'''

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))

    angle_derivs_df = pd.DataFrame()

    for action in action_nos:

        for subject in subject_nos:

            angle_df = df.loc[
                (df['action'] == action) & (df['subject'] == subject)]

            for deriv in [1,2]:

                window_size = 5
                polyorder = 2
                angle_df = compute_angle_derivs(angle_df, window_size, polyorder, deriv)

            angle_derivs_df = angle_derivs_df.append(angle_df)

    return angle_derivs_df

def control_display_derivs(df):
    '''Plots the deriv of the angles computed'''

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    action_type = ['Drinking', 'Eating', 'Reading', 'Phoning', 'Writing', 'Laptop', 'Cheering', 'Nothing', 'Throwing a ball', 'Playing a video game', 'Playing a guitar']

    for action, label in zip(action_nos, action_type):
            
        fig, axs = plt.subplots(3, 3)
        fig.suptitle(f'Action: {action} - {label}')

        for subject in subject_nos:

            angle_df = df.loc[
                (df['action'] == action) & (df['subject'] == subject)]

            axs[0, 0].plot(angle_df['back_angle'])
            axs[0, 0].set_ylabel('Angle (deg)')
            axs[0, 0].set_title('Back')
            axs[0, 1].plot(angle_df['left_angle'])
            axs[0, 2].plot(angle_df['right_angle'])

            deriv = 1

            axs[1, 0].plot(angle_df[f'back_{deriv}der'])
            axs[1, 1].plot(angle_df[f'left_{deriv}der'])
            axs[1, 0].set_ylabel('1st Deriv Angle (deg/s)')
            axs[0, 1].set_title('Left')
            axs[1, 2].plot(angle_df[f'right_{deriv}der'])

            deriv = 2

            axs[2, 0].plot(angle_df[f'back_{deriv}der'])
            axs[2, 1].plot(angle_df[f'left_{deriv}der'])
            axs[2, 2].plot(angle_df[f'right_{deriv}der'], label=f'Subject {subject}')
            axs[2, 0].set_ylabel('2nd Deriv Angle (deg/s^2)')
            axs[0, 2].set_title('Right')

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'center right')

    plt.show()

def main():
    '''Run the script'''

    # Set constants and settings
    FILE_DIR = '../patient-simulator-FYP/'
    DIAGRAM_DIR = FILE_DIR + 'diagrams/'
    DATA_DIR = FILE_DIR + 'datasets/'
    CONTROL_DIR = DATA_DIR + 'control/'
    plot_on = True
    save_new = False

    # Extract the 3D coords from relevant .txt files
    _ = txt_extract_and_filter(CONTROL_DIR)

    # Compute the angles from the corresponding files
    control_angles_df = coords_to_angles(CONTROL_DIR)

    # Resample the data to match patient data
    control_resamp_angles_df = resampler(control_angles_df, 30, 20)

    if plot_on:
        # Plot sets of control actions
        plot_control_actions(control_resamp_angles_df, DIAGRAM_DIR, save_new=save_new)

    # Normalise angles to preprocess before windowing
    control_norm_df = control_normaliser(control_resamp_angles_df)

    if plot_on:
        # Plot normalised angles
        plot_control_norm_data(control_norm_df, DIAGRAM_DIR, save_new=save_new)

    # Save a copy of preprocessed data as .csv
    save_proc_control_df(control_norm_df, CONTROL_DIR)

    ### NO LONGER INCLUDED IN THE PIPELINE ###
    ## control_derivs_df = control_angle_derivatives(control_norm_df)
    ## if plot_on:
    ##     control_display_derivs(control_derivs_df)

if __name__ == "__main__":
    main()
