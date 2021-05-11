# Patient-Behaviour-Detection

Patient-Behaviour-Detection is a demonstration jupyter notebook and supporting python scripts for a final year university project titled:

## Unsupervised Anomaly Detection of Multivariate Medical Sensor Data
---
The continued development of dynamic seating solutions for young children suffering from dystonic cerebral palsy has meant that the ability to understand and locate periods of whole-body extensor spasm within existing unlabelled multivariate sensor data is essential for improving chair designs.

This project aims to develop various versions of an unsupervised anomaly detection system which can be used for the purposes of the offline labelling of anomalies which may equate to less obvious indicators of spasm or other type of unusual behaviour.
## Installation
---
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-3710/)
1. Python 3.7 or later to run.
2. Create and activate a python virtual environment e.g. virtualenv, pyenv, poetry etc.
3. Run:
```bash
pip install -r requirements.txt  
```
## Usage
---
For data privacy reasons, the .csv patient data will not be uploaded to GitHub but if access is available to it, it should be placed within **`datasets/patient/DE250053 - (Callibrated - new method - compressed).csv`**. 
The control data is selected from the seated actions dataset [**MSRActionDaily3D**](https://sites.google.com/view/wanqingli/data-sets/msr-dailyactivity3d).

There are two main options for interacting with the project:
1. The main method is to examine **`anomaly_detection.ipynb`** which fully details the methodology followed:
    - It allows for the demonstration of the system with the default selected settings.
    - It allows for changes to be made to the *PARAMETERS* and for the notebook to then be rerun such as:
      - Run Settings: PLOT_ON, DIAGRAM_SAVE
      - Model Parameters: WINDOW_SIZE, TRAIN_DATA, COMPRESS_PER_LAYER
      - Detection Parameters: FIXED_THRESHOLDS_PCT, STD_MULTIPLIER, THETA
   
2. The two preprocessing scripts can be run on their own to produce two preprocessed .csv files prior to window sequencing. Ideally the whole operation would have been developed into a pipeline but this was not possible within the time constraints of this project.
```bash
python patient_data.py
python control_data.py
```
## Contributing
As basic supporting work for a university project, this repository is optimised for producing deliverables for a theoretical discussion rather than operational performance. Therefore, pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## License
[MIT](https://choosealicense.com/licenses/mit/)
