# Imports
from scipy.io import wavfile
import scipy.signal as sps
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test():
    # Your new sampling rate
    new_rate = 20
    old_rate = 30

    # Read file
    df = pd.read_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/angles_skeleton_data.csv',index_col=False)

    test_df = df[['back_angle', 'left_angle', 'right_angle']].loc[(df['action'] == 3) & (df['subject'] == 3)]

    # print(test_df.head())

    data = np.array(test_df['back_angle'].tolist())
    x = np.linspace(0, 10, len(data), endpoint=False)

    # Resample data
    number_of_samples = round(len(data) * float(new_rate) / old_rate)
    new_data = sps.resample(data, number_of_samples)
    xre = np.linspace(0, 10, len(new_data), endpoint=False)


    window_size = 5
    smoothed_data = sps.savgol_filter(new_data, window_size, 3)

    fig, axs = plt.subplots(3)

    fig.suptitle('Vertically stacked subplots')

    axs[0].plot(x,data)
    axs[1].plot(xre,new_data)
    axs[2].plot(xre,smoothed_data)
    plt.show()

def main():
    # Your new sampling rate
    new_rate = 20
    old_rate = 30

    # Read file
    df = pd.read_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/angles_skeleton_data.csv',index_col=False)

    # Loop through each action and subject
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    angles = ['back_angle', 'left_angle', 'right_angle']

    resamp_headers = ['action', 'subject', 'frame', 'back_angle', 'left_angle', 'right_angle']

    resamp_df = pd.DataFrame(columns=resamp_headers)

    for action in action_nos:

        for subject in subject_nos:

            angle_df = df[angles].loc[(df['action'] == action) & (df['subject'] == subject)]

            for angle in angles:

                # Extract data
                data = np.array(angle_df[angle].tolist())

                # Resample data
                number_of_samples = round(len(data) * float(new_rate) / old_rate)
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

    print(f'\nPrevious 30Hz DataFrame:\n')
    print(df)
    print(f'\nResampled 20Hz DataFrame:\n')
    print(resamp_df)

    # Output to csv
    resamp_df.to_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/resampled_angles_skeleton_data.csv', index=False)

if __name__ == "__main__":
    main()
