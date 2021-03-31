import pandas as pd 
import matplotlib.pyplot as plt
import os.path
import numpy as np

from control_data import compute_angle_derivs

def angle_derivs(angle_df):
    '''Script to calculate the deriv of the angles computed'''

    # # Read file
    # file_path = '/Users/jamesmeyer/Projects/fyp-skeleton-data/'
    # file_name = 'resampled_angles_skeleton_data.csv'
    # df = pd.read_csv(f'{file_path}{file_name}', index_col=False)

    angle_derivs_df = pd.DataFrame()

    for deriv in [1,2]:

        window_size = 5
        polyorder = 2
        angle_df = compute_angle_derivs(angle_df, window_size, polyorder, deriv)

    angle_derivs_df = angle_derivs_df.append(angle_df)

    print(angle_derivs_df)

    return angle_derivs_df

def data_extractor(file_path):

    file_name = 'DE250053 - (Callibrated - new method - compressed).csv'

    df = pd.read_csv(f'{file_path}{file_name}', index_col=False, skiprows=[0,1,2,4])

    print(df)

    return df

def data_filterer(df):

    keep_cols = ["Backrest Angle / Deg", "Left Hip Angle / Deg", "Right Hip Angle / Deg"]

    angle_df = df[keep_cols]

    angle_df = angle_df.rename(columns={
        "Backrest Angle / Deg": "back_angle", 
        "Left Hip Angle / Deg": "left_angle", 
        "Right Hip Angle / Deg": "right_angle"
        })

    print(angle_df)

    return angle_df

def format_to_control(angle_derivs_df, d_file_path, save_on):

    extra_cols = ["action", "subject", "frame"]

    extra_cols_df = pd.DataFrame(columns=extra_cols)

    angle_derivs_df = extra_cols_df.append(angle_derivs_df, ignore_index=True)

    print(angle_derivs_df)

    if save_on:
        new_file_name = 'patient_data.csv'
        if os.path.exists(f'{d_file_path}{new_file_name}'):
            print("csv created already, delete file if new version required")
        else:
            angle_derivs_df.to_csv(f'{d_file_path}{new_file_name}',index=False)
            print(f'Saved a copy as csv in :{d_file_path}')

    return angle_derivs_df

def display_derivs(angle_derivs_df):
    fig, axs = plt.subplots(3, 3)

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



def main():

    # path to local storage directory
    source_file_path = '/Users/jamesmeyer/University of Bath/Patient Simulator FYP - General/datasets/patient/dr-adlams-data/'
    
    df = data_extractor(source_file_path)

    angle_df = data_filterer(df)

    angle_derivs_df = angle_derivs(angle_df)

    display_derivs(angle_derivs_df)

    # Read file
    dest_file_path = '/Users/jamesmeyer/University of Bath/Patient Simulator FYP - General/datasets/patient/'

    control_df = format_to_control(angle_derivs_df, dest_file_path, True)


if __name__ == "__main__":
    main()

# /Users/jamesmeyer/University\ of\ Bath/Patient\ Simulator\ FYP\ -\ General/precursor-work/Seat\ movement\ and\ torque\ data/DE250053\ -\ \(Callibrated\ -\ new\ method\ -\ compressed\).csv 
