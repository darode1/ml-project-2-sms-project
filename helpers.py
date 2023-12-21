import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,TensorDataset
from torch.optim import lr_scheduler
import pandas as pd
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F

def pad_sequences(X, block_length, video_indices):
    '''
    Pads zero sequences to every sequence in an input tensor so that the
    length of every sequence is a multiple of block_length.

    Args:
        X (torch.Tensor): Input tensor containing sequences.
        block_length (int): Desired multiple for sequence length.
        video_indices (numpy.ndarray): Array of indices where each video starts.

    Returns:
        torch.Tensor: Padded tensor containing sequences.

    '''
    padded_sequences = []

    for i in range(len(video_indices) - 1):
        start_idx = video_indices[i]
        end_idx = video_indices[i + 1]
        sequence = X[start_idx:end_idx]

        # If the sequence is 1D, reshape it to 2D
        if len(sequence.shape) == 1:
            sequence = sequence.view(-1, 1)

        # Calculate the padding length
        padding_length = block_length - len(sequence) % block_length

        # Pad the sequence with zeros
        padded_sequence = F.pad(sequence, (0, 0, 0, padding_length), value=0)

        padded_sequences.append(padded_sequence)

    return torch.cat(padded_sequences, dim=0)


def import_data_exercise(train_df, test_df, exercise_mapping, augmented):
    '''
    Produces the train set and test set as tensors when predicting exercise.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        exercise_mapping (dict): Mapping of exercise labels to integers.
        augmented (bool): Flag indicating whether the data is augmented.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test as tensors.
    '''

    # Filter data based on augmentation
    if not augmented:
        train_df = train_df[train_df['video_id'] < 2100]

    # Extract features and labels from DataFrames
    feature_columns = ['Exercise', 'Set', 'Participant', 'Camera', 'video_id',
                       'encoded_exo_Abduction', 'encoded_exo_Bird', 'encoded_exo_Bridge',
                       'encoded_exo_Knee', 'encoded_exo_Shoulder', 'encoded_exo_Squat', 'encoded_exo_Stretch']

    X_train = train_df.drop(feature_columns, axis=1).astype('float32')
    X_test = test_df.drop(feature_columns, axis=1).astype('float32')

    y_train = train_df[['Exercise']]
    y_test = test_df[['Exercise']]

    # Convert DataFrames to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor([exercise_mapping[y[0]] for y in y_train.values], dtype=torch.int64)

    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor([exercise_mapping[y[0]] for y in y_test.values], dtype=torch.int64)

    return X_train, X_test, y_train, y_test


def import_data_set(train_df, test_df, set_mapping, augmented):
  '''
    Produces the train set and test set as tensors to predict set.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        set_mapping (dict): Mapping of set labels to integers.
        augmented (bool): Flag indicating whether to consider the augmented data.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test as tensors.
    '''
  if not augmented:
    train_df = train_df[train_df['video_id'] < 2100]

  X_train = train_df.drop(['Exercise', 'Set', 'Participant', 'Camera', 'video_id'], axis=1).astype('float32')
  X_test = test_df.drop(['Exercise', 'Set', 'Participant', 'Camera', 'video_id'], axis=1).astype('float32')

  y_train = train_df[['Set']]
  y_test = test_df[['Set']]

  X_train = torch.tensor(X_train.values, dtype=torch.float32)
  y_train = torch.tensor([set_mapping[y[0]] for y in y_train.values], dtype=torch.int64)

  X_test = torch.tensor(X_test.values, dtype=torch.float32)
  y_test = torch.tensor([set_mapping[y[0]] for y in y_test.values], dtype=torch.int64)
  return X_train, X_test, y_train, y_test

def import_data_exercise_set(train_df, test_df, exercise_set_mapping, augmented):
  '''
    Produces the train set and test set as tensors to predict both exercice and set.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        set_mapping (dict): Mapping of (exercise,set) tuples labels to integers.
        augmented (bool): Flag indicating whether to consider the augmented data.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test as tensors.
    '''
  if not augmented:
    train_df = train_df[train_df['video_id'] < 2100]

  X_train = train_df.drop(['Exercise', 'Set', 'Participant', 'Camera', 'video_id','encoded_exo_Abduction','encoded_exo_Bird', 'encoded_exo_Bridge', 'encoded_exo_Knee', 'encoded_exo_Shoulder', 'encoded_exo_Squat', 'encoded_exo_Stretch'], axis=1).astype('float32')
  X_test = test_df.drop(['Exercise', 'Set', 'Participant', 'Camera', 'video_id','encoded_exo_Abduction','encoded_exo_Bird', 'encoded_exo_Bridge', 'encoded_exo_Knee', 'encoded_exo_Shoulder', 'encoded_exo_Squat', 'encoded_exo_Stretch'], axis=1).astype('float32')

  y_train = train_df[['Exercise','Set']]
  y_test = test_df[['Exercise','Set']]

  X_train = torch.tensor(X_train.values, dtype=torch.float32)
  y_train = torch.tensor(np.array([exercise_set_mapping[tuple(y)] for y in y_train.iloc[:,:2].values]), dtype=torch.int64)

  X_test = torch.tensor(X_test.values, dtype=torch.float32)
  y_test = torch.tensor(np.array([exercise_set_mapping[tuple(y)] for y in y_test.iloc[:,:2].values]), dtype=torch.int64)
  return X_train, X_test, y_train, y_test

def test_accuracy(model, X_test, y_test, video_indices, device='cpu'):
    '''
    Prints the accuracy and scores of the model on a test set.

    Args:
        model (torch.nn.Module): The trained model.
        X_test (torch.Tensor): Test data features.
        y_test (torch.Tensor): Test data labels.
        video_indices (numpy.ndarray): Array of indices indicating the start of each video.
        device (str): Device to run the evaluation on (default is 'cpu').

    Returns:
        None
    '''
    model.to(device)
    model.eval()

    num_samples = X_test.size(0)

    # Move X_test and y_test to the same device as the model
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Make predictions on the test set
    with torch.no_grad():
        predictions_list = []

        # Process the data in batches to save memory
        batch_size = 16
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_X = X_test[start_idx:end_idx]

            # Make predictions on the current batch
            batch_predictions = model(batch_X)
            predictions_list.append(batch_predictions)

        # Concatenate predictions from all batches
        predictions = torch.cat(predictions_list)

    # Convert predictions to class labels
    predicted_labels = torch.argmax(predictions, dim=1)

    # Convert tensors to numpy arrays
    y_test_np = y_test.cpu().numpy()  # Move back to CPU for numpy conversion
    predicted_labels_np = predicted_labels.cpu().numpy()

    accuracy = accuracy_score(y_test_np, predicted_labels_np)
    print(f'Accuracy on each frame: {accuracy:.4f}')

    combined_array = np.column_stack((predicted_labels_np, y_test_np))

    y_videos = []
    pred_videos = []
    for i in range(len(video_indices) - 1):
        segment = combined_array[video_indices[i]:video_indices[i + 1]]
        y_video = segment[0][1]
        pred_video = np.argmax(np.bincount(segment[:, 0]))
        y_videos.append(y_video)
        pred_videos.append(pred_video)

    accuracy_video = accuracy_score(y_videos, pred_videos)
    print(f'Accuracy for videos: {accuracy_video:.4f}')

    print('Classification Report:')
    print(classification_report(y_videos, pred_videos, labels=np.unique(y_test_np)))



def test_accuracy_on_frame(model,X_test,y_test):
  '''
    Prints the accuracy and scores of the model on a test set.

    Args:
        model (torch.nn.Module): The trained model.
        X_test (torch.Tensor): Test data features.
        y_test (torch.Tensor): Test data labels.
        
    Returns:
        None
  '''
  model.eval()

  # Make predictions on the test set
  with torch.no_grad():
    predictions = model(X_test)

  # Convert predictions to class labels
  predicted_labels = torch.argmax(predictions, dim=1)

  # Convert tensors to numpy arrays
  y_test_np = y_test.numpy()
  predicted_labels_np = predicted_labels.numpy()

  # Print accuracy
  accuracy = accuracy_score(y_test_np, predicted_labels_np)
  print(f'Accuracy on each frame: {accuracy:.4f}')

  # Print classification report
  print('Classification Report for each frame:')
  print(classification_report(y_test_np, predicted_labels_np, labels=y_test.unique()))

def train_model(num_epochs, model, optimizer, criterion, dataloader, scheduler, save_path, device='cpu'):
    '''
    Trains a neural network model using the specified parameters.

    Args:
        num_epochs (int): Number of training epochs.
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        save_path (str): Path to save the trained model state dictionary.
        device (str): Device to run the training on (default is 'cpu').

    Returns:
        None
    '''
    for epoch in range(num_epochs):
        model.to(device)
        losses = []
        model.train()

        for idx, batch in enumerate(dataloader):
            print(f"\rProcessing: {100 * idx / len(dataloader):.2f}%", end='', flush=True)
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        torch.save(model.state_dict(), save_path)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(losses):.4f}')
