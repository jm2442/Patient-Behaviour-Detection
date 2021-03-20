import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

print('running')


path = '/Users/jamesmeyer/Projects/fyp-skeleton-data/'

file_name = 'a01_s01_e01_skeleton.txt'

file_object = open(f'{path}{file_name}', 'r').readlines()

# print(file_object[0])

frame_per_rows = []
frame = []
frame_count = -1
# list_of_lines = []
for line in file_object:

    # print(line)
    stripped_line = line.strip()
    line_list = stripped_line.split()
    # list_of_lines.append(line_list)

    if line_list[0] == '40':
        frame_count += 1
        # print(frame)
        frame_per_rows.append(frame)
        frame = []
    elif frame_count > -1:
        if line_list[3] == '1':
            line_list = list(map(float, line_list))
            frame.append(line_list[0:3])

# for frame in frame_per_rows:
#     print(frame)
# x_plots = [x[0] for x in frame_per_rows[1]]
# y_plots = [x[1] for x in frame_per_rows[1]]
# z_plots = [x[2] for x in frame_per_rows[1]]
indexes = [0, 1, 2, 12, 13, 16, 17]
snapshat_filtered = [frame_per_rows[2][x] for x in indexes]
snapshot = np.array(snapshat_filtered)

# snapshot = np.array(frame_per_rows[2])
print(snapshot)
# print(snapshot[:,1])



plt.scatter(snapshot[:,0], snapshot[:,1])
plt.axis('scaled')

plt.show()



# df = pd.read_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/a01_s01_e01_skeleton.txt')
# # patient-simulator-FYP
# print(df.head())


