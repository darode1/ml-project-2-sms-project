# Enhancing Home Physiotherapy: Machine Learning for Exercise Recognition and Error Detection

Our project builds upon David Rode’s research in Markerless Motion Capturing and Pose Estimation for therapeutic
applications.

## Project Description

There are three main goals in the project. The first being to correctly guess the performed exercises based on joints positions moving over time. The second being to detect potential mistakes in the executions of these exercises. The mistakes are called "Set" in our dataset, because they correspond to sets of mistakes per exercise. The third goal is to investigate influencing factors on prediction errors.

## Project structure

The repository is organized as follows. You can find descriptions of each section below.
* A notebook folder, where you will find all notebooks described below.
* A model folder, where you will find the main trained models
* A helpers.py with few methods to make notebooks clearer.

## Downloads

If you want to reproduce our results, you can download the following datasets through the link. There is no need to download everything at once, instead you can download each parquet file independently, depending of what you want to reproduce (more will be told on what to download below).

Download link: https://mega.nz/folder/VhsjCASI#9hjhq6UmYEEGNUsl12ZThg

## Reproducing the models

We will describe how to reproduce each of our results for each method. If you are only interested in our best result, you can only follow the first three subsections.

### Preprocessing

If you want to reproduce our preprocessing, you should download the file "dataset.parquet", and follow the steps of this section. Otherwise you can directly download the preprocessed datasets called "train_set_augmented.parquet" & "test_set.parquet".

To reproduce our preprocessing:
* Open the notebook preprocessing.ipynb
* Change the local_path variable to the path of "dataset.parquet"
* Run every cell

This will:
* Crop first 2 seconds of each video, and last second of the videos.
* Erase rows consisting of more than 20 consecutive Nans, as explained in the report.
* Linearly interpolate rows consisting of less the 20 consecutive Nans, to predict these missing values.
* Split the dataset into unbiased train and test sets. 20% of participants are taken in test set.
* Perform data augmentation of train set to rebalance the train set, and make it robut to noisy data.


### Exercise predictions
For the first task of predicting exercises 


### Error set predictions


#### Our best model for predicting error sets


#### CNNs

To reproduce this model:
* Open CNN_training.ipynb notebook
* Change the train_path and test_path variables with your own paths of "test_set.parquet" and "train_set_augmented.parquet".
* If you want to train on non-augmented data, you should change is_augmented to False. Otherwise, don't touch it.
* Depending of what CNN model you want to train, you might need to change the model variable from get_cnn2b() to get_cnn3b()
* Run every cell, and it will train the model.


#### RNN/LSTM

To reproduce this model:
* Open RNN_LSTM.ipynb notebook
* Replace train_path variable by the path for "train_set_augmented.parquet", and the test path by "test_set.parquet".
* Run the notebook, the test accuracy will be printed in the last cell.


#### Decision trees

### Ethical study

To reproduce the results of this part, you will need to download "slow_test.parquet", "fast_test.parquet" and "ethics.path"

* Open the ethics_MLP.ipynb notebook
* Start by changing the paths of model_path to the path of ethics.path
* Change paths of test_path to the test path, train_path to augmented train path, test_slow_path and test_fast_path to the paths of the newly downloaded csv.
* Run every cell, accuracies will be displayed at the end.

### External libraries used:
* Numpy
* Pandas (for reading our datas and performing minor processing)
* Seaborn/Matplotlib (for plotting purposes) 
* Torch (for modeling neural networks)
* Keras (used for modeling the LSTM model, saved a lot of coding time) 

## Contributors

- Baptiste Maquignaz [Baptiste-ic](https://github.com/Baptiste-ic)
- Garik Sahakyan [garikSahakayan](https://github.com/garikSahakayan)
- Rami Atassi [RamiATASSI](https://github.com/RamiATASSI)