# Enhancing Home Physiotherapy: Machine Learning for Exercise Recognition and Error Detection

Our project builds upon David Rode’s research in Markerless Motion Capturing and Pose Estimation for therapeutic
applications.

## Project Description

There are three main goals in the project. The first being to correctly guess the performed exercises based on joints positions moving over time. The second being to detect potential mistakes in the executions of these exercises. The mistakes are called "Set" in our dataset, because they correspond to sets of mistakes per exercise. The third goal is to investigate influencing factors on prediction errors.

## Repository Organization

The repository is organized as follows. You can find descriptions of each section below.
(à faire)

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

External libraries used:
* Numpy
* Pandas (for preprocessing purposes)
* Seaborn/Matplotlib (for plotting purposes) 


### Exercise predictions


### Error set predictions


#### Our best model for predicting error sets


#### CNNs


#### RNN/LSTM

To reproduce this model:
* Open RNN_LSTM.ipynb notebook
* Replace train_path variable by the path of "train_set_augmented.parquet", and the test path by "test_set.parquet".
* Run the notebook, the test accuracy will be printed in the last cell.

External libraries used:
* Numpy
* Pandas (for reading our datas and performing minor processing)
* Keras (used for modeling the LSTM model, saved a lot of coding time) 


#### Decision trees

### Ethical study

To reproduce the results of this part, you will need to download "slow_test.parquet", "fast_test.parquet" and "ethics.path"

* Open the ethics_MLP.ipynb notebook
* Start by changing the paths of model_path to the path of ethics.path
* Change paths of test_path to the test path, train_path to augmented train path, test_slow_path and test_fast_path to the paths of the newly downloaded csv.
* Run every cell, accuracies will be displayed at the end.

## Contributors

- Baptiste Maquignaz [Baptiste-ic](https://github.com/Baptiste-ic)
- Garik Sahakyan [garikSahakayan](https://github.com/garikSahakayan)
- Rami Atassi [RamiATASSI](https://github.com/RamiATASSI)