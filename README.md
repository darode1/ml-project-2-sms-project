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


### Model for predicting exercises


### Our best model for predicting error sets


### CNNs


### RNN/LSTM

For training this result, you will need to download "train_set_less_noisy.parquet". This is a similar version of the previous train set but with less noise on new inputs when performing data augmentation (to reproduce it, you can redo step 1 on preprocessing, but by changing noise_std variable to 0.01). Indeed, by crossvalidating, we found out that LSTM was less accurate when trained on data that was augmented with a lot of noise.

To reproduce this model:
* Replace train_path variable by the path of "train_set_less_noisy.parquet", and the test path by "test_set.parquet".
* Run the notebook, the test accuracy will be printed in the last cell.

External libraries used:
* Numpy
* Pandas (for reading our datas and performing minor processing)
* Keras (used for modeling the LSTM model, saved a lot of coding time) 


### Decision trees

## Ethical study

## Contributors

- Baptiste Maquignaz [Baptiste-ic](https://github.com/Baptiste-ic)
- Garik Sahakyan [garikSahakayan](https://github.com/garikSahakayan)
- Rami Atassi [RamiATASSI](https://github.com/RamiATASSI)