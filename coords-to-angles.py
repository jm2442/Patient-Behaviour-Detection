import pandas as pd 
import numpy as np
from ast import literal_eval
import math

def vectors_to_angle(vec1, vec2):

    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)

    return (angle/math.pi) * 180

def main():

    # Get csv file
    df = pd.read_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/extracted_filtered_skeleton_data.csv', index_col=False)

    back_angle = []
    left_angle = []
    right_angle = []

    # loop through each row
    for index, row in df.iterrows():

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
      
    # Add columns to df
    df['back_angle'] = back_angle
    df['left_angle'] = left_angle
    df['right_angle'] = right_angle

    print(df.head())

    df.to_csv('/Users/jamesmeyer/Projects/fyp-skeleton-data/angles_skeleton_data.csv',index=False)


if __name__ == "__main__":
    main()
