import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def number_to_string(num_list):

    str_list = []
    for num in num_list:
        if num < 10:
            str_list.append('0'+str(num))
        else:
            str_list.append(str(num))

    return str_list

print('running')

action_nos = list(range(1,17))
subject_nos = list(range(1,11))

path = '/Users/jamesmeyer/Projects/fyp-skeleton-data/'

for action in number_to_string(action_nos):

    for subject in number_to_string(subject_nos):

        file_name = f'a{action}_s{subject}_e01_skeleton.txt'

file_object = open(f'{path}{file_name}', 'r').readlines()


frame_per_rows = []
frame = []
frame_count = -1
for line in file_object:

    stripped_line = line.strip()
    line_list = stripped_line.split()

    if line_list[0] == '40':

        frame_count += 1
        frame_per_rows.append(frame)
        frame = []

    elif frame_count > -1:

        if line_list[3] == '1':

            line_list = list(map(float, line_list))
            frame.append(line_list[0:3])

samples = []
for sample in frame_per_rows:

    indexes = [0, 1, 2, 12, 13, 16, 17] # points relating to the placement of points
    snapshot_filtered = [frame_per_rows[2][x] for x in indexes]
    samples.append(snapshot_filtered)

snapshot = np.array(samples)
# print(snapshot)

# plt.scatter(snapshot[:,0], snapshot[:,1])
# plt.axis('scaled')

# plt.show()



# df = pd.read_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/a01_s01_e01_skeleton.txt')
# # patient-simulator-FYP
# print(df.head())
# 

if __name__ == "__main__":
    pass

