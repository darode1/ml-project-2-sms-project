# Enhancing Home Physiotherapy: Machine Learning for Exercise Recognition and Error Detection

## Project Overview

Building upon David Rode’s research in Markerless Motion Capturing and Pose Estimation for therapeutic applications.

### Project Goals

1. Correctly guess performed exercises based on joint positions over time.
2. Detect potential mistakes in exercise executions.
3. Investigate influencing factors on prediction errors.

## Project structure

The repository is organized as follows. You can find descriptions of each section below.
* A notebook folder, where you will find all notebooks described below.
* A pre-trained models folder named paths, where you will find the main trained models
* A helpers.py with few methods to make notebooks clearer.


## Downloads

Reproduce results by downloading datasets from [this link](https://mega.nz/folder/x58iACqb#EciOgNhfLUL30SUhvxe6gw).

## Reproducing the Models

We will describe how to reproduce our results for the main models. If you are only interested in our best result, you can ignore the following subsections: Other MLP models, CNNs, RNN/LSTM and Decision trees.

### Preprocessing

If you want to reproduce our preprocessing, you should download the file "dataset.parquet", and follow the steps of this section. Otherwise you can directly download the preprocessed datasets called "train_set_augmented.parquet" & "test_set.parquet".

- Open the notebook **preprocessing.ipynb**.
- Change the `local_path` variable to the path of **dataset.parquet.**
- Run every cell

This will:
* Crop first 2 seconds of each video, and last second of the videos.
* Erase rows consisting of more than 20 consecutive Nans, as explained in the report.
* Linearly interpolate rows consisting of less the 20 consecutive Nans, to predict these missing values.
* Split the dataset into unbiased train and test sets. 20% of participants are taken in test set.
* Perform data augmentation of train set to rebalance the train set, and make it robut to noisy data.

### Exercise predictions
For this part, you will need to download "test_set.parquet" and "train_set_augmented.parquet".

For the first task of predicting exercises, we used mlp3x256. To reproduce it:
- Open **Main_model.ipynb**.
- Change `train_path` and `test_path` to your path to **train_set_augmented.parquet** and **test_set.parquet**, respectively.
- Set `train_eval` to True for retraining, False for evaluation.
- Set `is_augmented` to False if you want to train our model without data augmentation in train set.
- Run cells in "First section" and "Training the model to predict _*exercises*_ to train or evaluate (print accuracies)

### Error Set Predictions

#### PhysioMLP - Best Model

For the second task, we created a model called PhysioMLP. To reproduce it:
- Open **Main_model.ipynb**.
- Change `train_path` and `test_path` to your path to **train_set_augmented.parquet** and **test_set.parquet**, respectively.
- Set `train_eval` to True for retraining, False for evaluation.
- Set `is_augmented` to False if you want to train our model without data augmentation in train set.
- Run cells in "First section" and "Training the model to predict _*sets*_" to train or evaluate (print accuracies)


#### Other MLP models

In this section we have many other MLP models that we tested, to reproduce them:
- Open **MLP_training.ipynb**.
- Change `train_path` and `test_path` to your path to **train_set_augmented.parquet** and **test_set.parquet**, respectively.
- Set `is_augmented` to False if you want to train our model without data augmentation in train set.
- Depending on what MLP model you want to train, you might need to change the model variable with your desired MLP model.
- Run every cell, and it will train the model.

##### CNNs

To reproduce this model:
- Open **CNN_training.ipynb**.
- Change `train_path` and `test_path` to your path to **train_set_augmented.parquet** and **test_set.parquet**, respectively.
- Set `is_augmented` to False if you want to train our model without data augmentation in train set.
- Depending on what CNN model you want to train, you might need to change the model variable from get_cnn2b() to get_cnn3b()
- Run every cell, and it will train the model.

#### RNN/LSTM

To reproduce this model:
- Open **RNN_LSTM.ipynb**.
- Change `train_path` and `test_path` to your path to **train_set_augmented.parquet** and **test_set.parquet**, respectively.
- Run the notebook, the test accuracy will be printed in the last cell.

#### Random Forest

To reproduce this model:
- Open **Random_Forest.ipynb**.
- Change `prep_path` to your path to **prep_dataset.parquet**.
- Change `train_path` and `test_path` to your path to **train_set_augmented.parquet** and **test_set.parquet**, respectively.
- Run cells in "Import and paths" and "Machine Learning" to both train and evaluate the model

#### Ethical Study

To reproduce the results of this part, you will need to download "slow_test.parquet", "fast_test.parquet" and "ethics.path"

- Open **ethics_MLP.ipynb**.
- Change paths for ethical study.
- Run every cell for displaying accuracies.

## External Libraries Used

| Library       | Version | Purpose                    |
|---------------|---------|----------------------------|
| Numpy         | 1.25.2  | Algebra                    |
| Pandas        | 2.0.3   | Data Manipulation          |
| Seaborn       | 0.12.2  | Plot                       |
| Matplotlib    | 3.7.2   | Plot                       |
| PyTorch       | 2.0.1   | Neural Networks            |
| Keras         | 2.10.0  | LSTM Model                 |
| Scikit-learn  | 1.3.0   | Random Forest and ML Classics|


## Contributors

- [Baptiste Maquignaz](https://github.com/Baptiste-ic)
- [Garik Sahakyan](https://github.com/garikSahakayan)
- [Rami Atassi](https://github.com/RamiATASSI)
