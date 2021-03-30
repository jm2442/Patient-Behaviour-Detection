'''A script to retrieve and prep the data into a form ready for preprocessing steps'''


from ast import literal_eval
import os.path
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps

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

def txt_extract_and_filter(s_file_path, d_file_path, save_on):
    '''Runs the entire script to extract and filter data from .txts'''

     # number of actions and subjects from the dataset
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))

    print("\nExtracting the 3D skeleton points for all of the different subjects' correct actions\n")

    # list to append dicts to form df
    samples = []

    # loop through actions and subjects
    for action in number_to_string(action_nos):

        for subject in number_to_string(subject_nos):

            # concat filename for extraction
            file_name = f'a{action}_s{subject}_e01_skeleton.txt'

            # Open file as list of lines
            file_object = open(f'{s_file_path}{file_name}', 'r').readlines()

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

    print("Raw extracted and filtered skeletal 3D points")
    print(samples_df)

    if save_on:
        new_file_name = 'coords_data.csv'
        if os.path.exists(f'{d_file_path}{new_file_name}'):
            print("csv created already, delete file if new version required")
        else:
            samples_df.to_csv(f'{d_file_path}{new_file_name}',index=False)
            print(f'Saved a copy as csv in :{d_file_path}')

    return samples_df

def vectors_to_angle(vec1, vec2):
    '''Takes to vectors and calculates the angle between them'''

    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)

    return (angle/math.pi) * 180

def coords_to_angles(file_path, save_on):
    '''Runs the entire script to convert 3D positions to angles'''

    file_name = 'coords_data.csv'
    df = pd.read_csv(f'{file_path}{file_name}', index_col=False)

    back_angle = []
    left_angle = []
    right_angle = []

    print("\nConverting the 3D skeleton points to the corresponding angles\n")

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
      
    # Add columns to df
    df[columns_add[0]] = back_angle
    df[columns_add[1]] = left_angle
    df[columns_add[2]] = right_angle

    # Transform the values from the obtuse to acute
    for col in columns_add:
        df[col] = 180 - df[col]

    print("Corresponding angles between skeleton coordinates")
    print(df)

    if save_on:
        new_file_name = 'angles_data.csv'
        if os.path.exists(f'{file_path}{new_file_name}'):
            print("csv created already, delete file if new version required")
        else:
            df.to_csv(f'{file_path}{new_file_name}',index=False)
            print(f'Saved a copy as csv in :{file_path}')

    return df

def resampler(old_hz, new_hz, file_path, save_on):
    '''Script to resample data from 30Hz to 20Hz'''

    file_name = 'angles_data.csv'
    df = pd.read_csv(f'{file_path}{file_name}',index_col=False)

    print(f"\nResampling the data from {old_hz}Hz to {new_hz}Hz \n")

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

            resamp_df = resamp_df.append(append_df, ignore_index = True)

    print(f"Resampled {new_hz}Hz corresponding angles between skeleton coordinates")
    print(resamp_df)

    if save_on:
        new_file_name = 'resampled_angles_data.csv'
        if os.path.exists(f'{file_path}{new_file_name}'):
            print("csv created already, delete file if new version required")
        else:
            resamp_df.to_csv(f'{file_path}{new_file_name}',index=False)
            print(f'Saved a copy as csv in :{file_path}')

    return resamp_df

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

def derivatives(file_path, save_on):
    '''Script to calculate the deriv of the angles computed'''

    file_name = 'resampled_angles_data.csv'
    df = pd.read_csv(f'{file_path}{file_name}', index_col=False)

    print(f"\nCalculating the first and second derivatives for the angles\n")

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

   
    print(f"First and second derivatives of angles")
    print(angle_derivs_df)

    if save_on:
        new_file_name = 'control_data.csv'
        if os.path.exists(f'{file_path}{new_file_name}'):
            print("csv created already, delete file if new version required")
        else:
            angle_derivs_df.to_csv(f'{file_path}{new_file_name}',index=False)
            print(f'Saved a copy as csv in :{file_path}')

    return angle_derivs_df

def display_deriv(df):
    '''Script to plot the deriv of the angles computed'''

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    angles = ['back_angle', 'left_angle', 'right_angle']
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

    # path to local storage directory
    source_file_path = '/Users/jamesmeyer/University of Bath/Patient Simulator FYP - General/datasets/control/sitting-skeleton-txts/'

    dest_file_path = '/Users/jamesmeyer/University of Bath/Patient Simulator FYP - General/datasets/control/'

    file_save = True

    filt_skel_coords_df = txt_extract_and_filter(source_file_path, dest_file_path, file_save)
    skel_angles_df = coords_to_angles(dest_file_path, file_save)
    resamp_angles_df = resampler(30, 20, dest_file_path, file_save)
    angles_df = derivatives(dest_file_path, file_save)
    # display_deriv(angles_df)

if __name__ == "__main__":
    main()
