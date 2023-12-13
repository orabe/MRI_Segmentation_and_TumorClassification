import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
import keras.backend as K

import pandas as pd
import matplotlib.pyplot as plt
import os


"""Compute metric between the predicted segmentation and the ground truth
"""

def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss


# ===============================================================
# ===============================================================
# ===============================================================

# define per class evaluation of dice coef
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

# ===============================================================
# ===============================================================
# ===============================================================

def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# ===============================================================
# ===============================================================
# ===============================================================

def plot_acc_loss_iou(show_plot=True, save_path=None):
    
    # Read the CSVlogger file that contains all our metrics (accuracy, loss, dice_coef, ...) of our training
    history = pd.read_csv('training.log', sep=',', engine='python')

    # Plot training and validation metrics
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(history['epoch'], history['accuracy'], 'b', label='Training Accuracy')
    axs[0].plot(history['epoch'], history['val_accuracy'], 'r', label='Validation Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(history['epoch'], history['loss'], 'b', label='Training Loss')
    axs[1].plot(history['epoch'], history['val_loss'], 'r', label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # axs[2].plot(history['epoch'], history['dice_coef'], 'b', label='Training dice coef')
    # axs[2].plot(history['epoch'], history['val_dice_coef'], 'r', label='Validation dice coef')
    # axs[2].set_xlabel('Epoch')
    # axs[2].set_ylabel('Dice Coefficient')
    # axs[2].legend()

    axs[2].plot(history['epoch'], history['mean_io_u'], 'b', label='Training mean IOU')
    axs[2].plot(history['epoch'], history['val_mean_io_u'], 'r', label='Validation mean IOU')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Mean IOU')
    axs[2].legend()

    # Add space between subplots
    plt.subplots_adjust(wspace=0.4)

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        fig_filename = 'plot_acc_loss_iou.png'
        fig_path = os.path.join(save_path, fig_filename)
        plt.savefig(fig_path)

        print(f"Figure saved at: {fig_path}")
    
    if show_plot:
        plt.show()
        

# ===============================================================
# ===============================================================
# ===============================================================

def plot_dice(show_plot=True, save_path=None):
        
    # Read the CSVlogger file that contains all our metrics (accuracy, loss, dice_coef, ...) of our training
    history = pd.read_csv('training.log', sep=',', engine='python')
    print(history.columns)
    # Plot training and validation metrics
    fig, axs = plt.subplots(1, 4, figsize=(16, 8))
    
    axs[0].plot(history['epoch'], history['dice_coef_necrotic'], 'b', label='Training')
    axs[0].plot(history['epoch'], history['val_dice_coef_necrotic'], 'r', label='Validation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('dice coef (necrotic)')
    axs[0].legend()
    
    axs[1].plot(history['epoch'], history['dice_coef_edema'], 'b', label='Training')
    axs[1].plot(history['epoch'], history['val_dice_coef_edema'], 'r', label='Validation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('dice coef (edema)')
    axs[1].legend()

    axs[2].plot(history['epoch'], history['dice_coef_enhancing'], 'b', label='Training')
    axs[2].plot(history['epoch'], history['val_dice_coef_enhancing'], 'r', label='Validation')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('dice coef (enhancing)')
    axs[2].legend()
    
    axs[3].plot(history['epoch'], history['dice_coef'], 'b', label='Training')
    axs[3].plot(history['epoch'], history['val_dice_coef'], 'r', label='Validation')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('dice coef (all 4)')
    axs[3].legend()

    # Add space between subplots
    plt.subplots_adjust(wspace=0.4)

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        fig_filename = 'plot_dice.png'
        fig_path = os.path.join(save_path, fig_filename)
        plt.savefig(fig_path)

        print(f"Figure saved at: {fig_path}")
    
    if show_plot:
        plt.show()

    
    
def print_save_eval(results, evaluation_path):
    descriptions = ["Loss", "Accuracy", "MeanIOU", 
                    "Dice coefficient",  
                    "Precision", "Sensitivity", "Specificity",
                    "dice_coef_necrotic",
                    "dice_coef_edema", "dice_coef_enhancin"]

    # Combine results list and descriptions list
    results_list = zip(results, descriptions)

    # Create a DataFrame
    df = pd.DataFrame(list(results_list), columns=["Metric", "Description"])

    # Round the Metric column to 4 decimal places
    df["Metric"] = df["Metric"].round(4)

    # Save DataFrame to CSV
    csv_file_path = os.path.join(evaluation_path, 'eval_metrics_results.csv')
    df.to_csv(csv_file_path, index=False)
    
    # Print the DataFrame
    # Display each metric with its description
    print("\nModel evaluation on the test set:")
    print("==================================")
    print(df)
    print("==================================")
    