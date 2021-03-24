'''Not sure yet'''
import pandas as pd
import matplotlib.pyplot as plt

def overall():
    '''Run entire script'''

    # Read file
    file_path = '/Users/jamesmeyer/Projects/fyp-skeleton-data/'
    file_name = 'resampled_angles_skeleton_data.csv'
    df = pd.read_csv(f'{file_path}{file_name}', index_col=False)

    # Loop through each action and plot the angles
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))
    angles = ['back_angle', 'left_angle', 'right_angle']
    action_type = ['Drinking', 'Eating', 'Reading', 'Phoning', 'Writing', 'Laptop', 'Cheering', 'Nothing', 'Throwing a ball', 'Playing a video game', 'Playing a guitar']

    for action, label in zip(action_nos, action_type):

        fig, axs = plt.subplots(3)

        fig.suptitle(f'Action: {action} - {label}')
        # fig.legend(subject_nos)

        for subject in subject_nos:

            angle_df = df[angles].loc[
                (df['action'] == action) & (df['subject'] == subject)]

            axs[0].plot(angle_df['back_angle'])
            axs[0].set_ylabel('Back Angle (Deg)')
            axs[1].plot(angle_df['left_angle'])
            axs[1].set_ylabel('Left Angle (Deg)')
            axs[2].plot(angle_df['right_angle'], label=f'Subject {subject}')
            axs[2].set_ylabel('Right Angle (Deg)')


        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'center right')

    plt.show()

if __name__ == "__main__":
    overall()
