# temperature-dependant

## Description
My version of the neural network created by Lu et al., used to find the impact of temperature.
Lu et al.'s paper can be found [here](https://www.pnas.org/content/early/2020/03/13/1922210117), and the code they used can be found [here](https://github.com/lululxvi/deep-learning-for-indentation).

## Data
This work uses different data than Lu et al. It can be found in the [data](data) folder. The data was collected from nanoindentation tests performed on samples of 33% TiAlTa at different temperatures.

## Code
All code is contained in the [src](src) folder. A short summary of each file is provided below:
- Factors in dataedit.py need to be changed depending on which temperature data is being adjusted. These are temperature, method, and n. This file is used to produce the neural network's input data.
- Figures used in presentations were created using figures.py.
- Writh runmultiple.py, multiple functions can be performed in parallel to speed up processing time.
- Fitting functions were used in fit_n.py. The yield stress and elastic stress files must be cleared before a new yield stress dataset can be created for a material and fit_n.py can be used.
- 

Besides conventional python packages, the following package by Lu Lu is required to run the programs.
- [DeepXDE](https://github.com/lululxvi/deepxde) v1.8.2 is used. Some DeepXDE functions may need to be modified if a different version is used.

## Questions
For help using this code, please see the issues section or open a new issue.