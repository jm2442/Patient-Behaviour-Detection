'''Script to find first and second derivatives of angle data'''
import scipy.signal as sps
import pandas as pd
import matplotlib.pyplot as plt

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

def angle_derivs_file():
    '''Script to calculate the deriv of the angles computed'''

    # Read file
    file_path = '/Users/jamesmeyer/Projects/fyp-skeleton-data/'
    file_name = 'resampled_angles_skeleton_data.csv'
    df = pd.read_csv(f'{file_path}{file_name}', index_col=False)

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

    # Output to csv
    file_name = 'angles_n_derivs_data.csv'
    angle_derivs_df.to_csv(f'{file_path}{file_name}', index=False)

    return angle_derivs_df
        

if __name__ == "__main__":

    angle_derivs_df = angle_derivs_file()

    display_deriv(angle_derivs_df)

