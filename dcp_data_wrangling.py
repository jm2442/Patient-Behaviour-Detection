import pandas as pd 
# import matplotlib.pyplot as plt
import numpy as np


def main():
    # number of actions and subjects from the dataset
    action_nos = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14]
    subject_nos = list(range(1,11))

    # path to local storage directory
    path = '/Users/jamesmeyer/University of Bath/Patient Simulator FYP - General/precursor-work/Seat movement and torque data/'
    
    file_name = 'DE250053 - (Callibrated - new method - compressed).csv'

    df = pd.read_csv(f'{path}{file_name}', index_col=False, skiprows=[0,1,2,4])

    print(df)
    print(df.dtypes)

    keep_cols = ["Backrest Angle / Deg", "Left Hip Angle / Deg", "Right Hip Angle / Deg"]

    angle_df = df[keep_cols]

    print(angle_df)
    
    angle_df = angle_df.rename(columns={
        "Backrest Angle / Deg": "back_angle", 
        "Left Hip Angle / Deg": "left_angle", 
        "Right Hip Angle / Deg": "right_angle"
        })

    print(angle_df)

if __name__ == "__main__":
    main()

# /Users/jamesmeyer/University\ of\ Bath/Patient\ Simulator\ FYP\ -\ General/precursor-work/Seat\ movement\ and\ torque\ data/DE250053\ -\ \(Callibrated\ -\ new\ method\ -\ compressed\).csv 
