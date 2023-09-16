# ISR_NAS


Neural Network Models for Ionospheric Electron Density Prediction: A Neural Architecture Search Study


## Prerequisites
Environment: Python 3, the key module is AutoKeras which can be referred [here][(https://autokeras.com/)]. 
It is suggested to install other modules based on AutoKeras, usually tensorflow 2 and Python 3.8 would be fine.

## Database
[ISR-NAS][(http://cedar.openmadrigal.org/)](http://cedar.openmadrigal.org/)
You can select the incoherent scatter radar you desire and customize the portion of data you want. Besides, the external geo-indices F10.7 and Ap3 are available in instrument categories under "Acess data".

The processed data of ISR and four model output for plotting are included in `./data/`, with ISR processed data saved as Pandas dataframe `*.lz4` and the model output as `*.npz`.

## To run the project
To run the AutoKeras search, refer to `runAK.py`, then `extract_ak.ipynb` to extract information. 
To run the manual neural network, refer to `dnn_ak.ipynb`.

## References
GitHub site of project [autokeras](https://github.com/keras-team/autokeras) by Haifeng Jin.
