import pandas as pd 
# import matplotlib.pyplot as plt
# import numpy as np

def number_to_string(num_list):
    "Basic function to aid with filename looping, converting single digit ints to 0 leading str versions"
    str_list = []
    for num in num_list:
        if num < 10:
            str_list.append('0'+str(num))
        else:
            str_list.append(str(num))

    return str_list


def main():
    # number of actions and subjects from the dataset
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))

    # path to local storage directory
    path = '/Users/jamesmeyer/Projects/fyp-skeleton-data/'

    # list to append dicts to form df
    samples = []

    # loop through actions and subjects
    for action in number_to_string(action_nos):

        for subject in number_to_string(subject_nos):

            # concat filename for extraction
            file_name = f'a{action}_s{subject}_e01_skeleton.txt'
            
            # For simple tracking purposes
            print(f'\nExtracting and filtering: {file_name}\n')

            # Open file as list of lines
            file_object = open(f'{path}{file_name}', 'r').readlines()

            frame_per_rows = []
            frame = []
            frame_count = -1

            # loop through lines and strip any trailing or leading whites spaces before splitting into list of lists
            for line in file_object:

                stripped_line = line.strip()
                line_list = stripped_line.split()

                # Each new frame is noted by a single line 40 so this is used as the identifier
                if line_list[0] == '40':
                    
                    # Prevent the initial two useless lines from each file causing an issue
                    if frame_count > -1:
                        frame_per_rows.append(frame)
                        frame = []

                    frame_count += 1

                elif frame_count > -1:
                    
                    # Issue with one file found a13-s06, so this is to notify that this one frame was disregarded.
                    try:

                        if line_list[3] == '1':

                            line_list = list(map(float, line_list))
                            frame.append(line_list[0:3])

                    except Exception as ex:
                        print(ex)
                        print(frame_count)
                        continue

            frame_no = 1
            # Loop through created list and convert info and extracted into dict to allow easy conversion to csv
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

    df = pd.DataFrame(samples)

    print(df.shape)

    df.to_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/extracted_filtered_skeleton_data.csv', index=False)

    # plt.scatter(snapshot[:,0], snapshot[:,1])
    # plt.axis('scaled')
    # plt.show()

if __name__ == "__main__":
    main()

