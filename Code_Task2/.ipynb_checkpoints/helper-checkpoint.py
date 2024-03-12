from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import subprocess
import csv

######################################################################## 
######################## Functions for Training ########################
########################################################################
def train_epoch(model, train_dl, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch on the training data.

    Args:
        model: The neural network model.
        train_dl: The training data loader.
        optimizer: The optimizer used for training.
        scheduler: Learning rate scheduler.
        criterion: The loss function.
        device: The device on which the model and data are located.

    Returns:
        Tuple[float, float]: Average training loss and accuracy for the epoch.
    """    
    # Initialize variables to track training loss and accuracy
    tr_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Create a tqdm progress bar for the training data loader
    epoch_iterator_train = tqdm(train_dl)
    
    # Iterate over batches in the training data loader
    for step, batch in enumerate(epoch_iterator_train):
        model.train()
        # Set the model to training mode
        torch.cuda.empty_cache()
         # Move batch data to the specified device
        images, targets = batch["image"].to(device), batch["target"].to(device)
        outputs = model(images)
        # Calculate the loss
        loss = criterion(outputs.squeeze(1), targets.float())
        # Calculate predictions and update accuracy metrics
        predictions = torch.round(torch.sigmoid(outputs))
        correct_predictions += (predictions == targets.view_as(predictions)).sum().item()
        total_samples += targets.size(0)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
        # Update training loss and display progress
        tr_loss += loss.item()
        epoch_iterator_train.set_postfix(batch_loss=(loss.item()), loss=(tr_loss / (step + 1)), accuracy=(correct_predictions / total_samples))
    # Adjust learning rate using scheduler
    scheduler.step()
    return tr_loss / len(epoch_iterator_train), correct_predictions / total_samples

def train_model(model, train_dl, validation_dl, optimizer, scheduler, criterion, n_epochs, device):
    """
    Trains the model for a specified number of epochs, performs validation, and saves the best model.

    Args:
        model: The neural network model.
        train_dl: The training data loader.
        validation_dl: The validation data loader.
        optimizer: The optimizer used for training.
        scheduler: Learning rate scheduler.
        criterion: The loss function.
        n_epochs: Number of training epochs.
        device: The device on which the model and data are located.
        save_dir: Directory to save the best model.

    Returns:
        Tuple[Dict[str, List[float]], torch.nn.Module, float]: Results dictionary, best model, and best threshold.
    """
    # Initialize best model and best AUC
    model.zero_grad()
    model.to(device)
    best_model = model
    best_thresh = 0
    best_i = 0
    best_train_loss, best_train_accuracy, best_val_loss, best_val_accuracy = 0, 0, 0, 0
    
    # Initialize results dictionary to store training and validation metrics
    results = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}
    for counter in range(n_epochs):
        # Train the model for one epoch
        train_loss, train_accuracy = train_epoch(model, train_dl, optimizer, scheduler, criterion, device)
        
        # Validate the model and get validation loss, accuracy, and best threshold
        val_loss, val_accuracy, best_thresh = validate_model(model, validation_dl, criterion, device)
        #val_loss, val_accuracy, best_thresh = train_loss, train_accuracy, 0.5
        
        # Print validation metrics
        print(f"EPOCH {counter+1}/{n_epochs}: \n  Validation loss: {val_loss}\n  Validation accuracy  = {val_accuracy}")
        
        # Save the model if the current validation accuracy is better than the previous best
        if val_accuracy > best_val_accuracy:
            best_i = counter
            best_model = model
            best_thresh = best_thresh
            best_train_loss, best_train_accuracy, best_val_loss, best_val_accuracy = train_loss, train_accuracy, val_loss, val_accuracy
        
        # Append training and validation metrics to the results dictionary
        results['train_losses'].append(train_loss)
        results['train_accuracies'].append(train_accuracy)
        results['val_losses'].append(val_loss)
        results['val_accuracies'].append(val_accuracy)
    return results, best_model, best_i, best_thresh, best_train_loss, best_train_accuracy, best_val_loss, best_val_accuracy

######################################################################## 
####################### Functions for Validation #######################
########################################################################
def validate_model(model, validation_dl, criterion, device):
    """
    Validates the model on the validation dataset and computes loss, AUC score, and best threshold.

    Args:
        model: The neural network model.
        validation_dl: The validation data loader.
        criterion: The loss function.
        device: The device on which the model and data are located.

    Returns:
        Tuple[float, float, float]: Validation loss, AUC score, and best threshold.
    """
    # Set the model to evaluation mode
    with torch.no_grad():
        model.eval()
        
        # Initialize variables for validation loss and prediction arrays
        val_loss = 0.0
        preds = []
        true_labels = []
        # Create a tqdm progress bar for the validation data loader
        epoch_iterator_val = tqdm(validation_dl)
        
        # Iterate over batches in the validation data loader
        for step, batch in enumerate(epoch_iterator_val):
            # Move batch data to the specified device
            images, targets = batch["image"].to(device), batch["target"].to(device)
            outputs = model(images)
            
            # Calculate the loss
            loss = criterion(outputs.squeeze(1), targets.float())
            val_loss += loss.item()
            
            # Update progress bar with current loss
            epoch_iterator_val.set_postfix(batch_loss=(loss.item()), loss=(val_loss / (step + 1)))
            
            # Collect predictions and true labels
            preds.append(outputs.sigmoid().detach().cpu().numpy())
            true_labels.append(targets.cpu().numpy())
            
        # Combine predictions and true labels
        preds = np.vstack(preds).T[0].tolist()
        true_labels = np.hstack(true_labels).tolist()
        
        # Calculate AUC score
        auc_score = roc_auc_score(true_labels, preds)
        
        # Initialize variables for best threshold and adjusted AUC score
        auc_score_adj_best = 0
        # Search for the best threshold
        for thresh in np.linspace(0, 1, 50):
            auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
            if auc_score_adj > auc_score_adj_best:
                best_thresh = thresh
                auc_score_adj_best = auc_score_adj
    return val_loss / len(epoch_iterator_val), auc_score, best_thresh

######################################################################## 
######################## Functions for Testing #########################
########################################################################
def get_params(file_path):
    """
    Read a text file containing model summary and extract relevant parameters.

    Args:
        file_path (str): Path to the text file.

    Returns:
        dict: Dictionary containing extracted model parameters.
    """
    model_summary = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        model_summary['num_3d_images'] = int(lines[4].split('=')[1].strip())
        model_summary['batch_size'] = int(lines[5].split('=')[1].strip())
        model_summary['learning_rate'] = float(lines[6].split('=')[1].strip().strip('[]'))
        model_summary['threshold'] = float(lines[18].split('=')[1].strip())
    return model_summary

def load_model(model, model_path):
    state_dict = torch.load(f"{model_path}/model_weights.pth")
    model.load_state_dict(state_dict)
    model_summary = get_params(f"{model_path}/model_summary.txt")
    return model, model_summary

def get_prediction(model, test_dl, device):
    """
    Get predictions from a model on a test dataloader.

    Args:
        model: The PyTorch model for predictions.
        test_dl: Dataloader for the test dataset.
        device: Device to which the model and data should be moved.

    Returns:
        dict: Dictionary containing predictions for each sample in the test dataset.
    """
    # Move the model to the specified device
    model.to(device)
    
    # Initialize a dictionary to store predictions
    prediction = {}
    
    with torch.no_grad(): 
        model.eval()
        # Initialize a list to store predictions for each batch
        preds = []
        
        epoch_iterator_val = tqdm(test_dl)
        for step, batch in enumerate(epoch_iterator_val):
            # Computing the prediction for one batch
            images = batch["image"].to(device)
            outputs = model(images)
            
            # Append predictions to the list
            #preds.append(outputs.sigmoid().detach().cpu().numpy())

            # Update the prediction dictionary with individual sample predictions
            for i in range(len(batch["id"])):
                prediction[batch["id"][i]] = outputs.sigmoid().detach().cpu().numpy()[i].item()
    return prediction

def most_voted_predictions(flair_predictions, T1w_predictions, T1wCE_predictions, T2w_predictions,
                           flair_thr, T1w_thr, T1wCE_thr, T2w_thr):
    vot_pred = {}
    for key in flair_predictions.keys():
        count = ( int(flair_predictions[key] > flair_thr ) + 
                  int(T1w_predictions[key]   > T1w_thr   ) +
                  int(T1wCE_predictions[key] > T1wCE_thr ) +
                  int(T2w_predictions[key]   > T2w_thr   ) )
        vot_pred[key] = (count > 2)
    return vot_pred

def average_predictions(flair_predictions, T1w_predictions, T1wCE_predictions, T2w_predictions):
    avg_pred = {}
    for key in flair_predictions.keys():
        avg_pred[key] = (flair_predictions[key] + T1w_predictions[key] 
                         + T1wCE_predictions[key] + T2w_predictions[key]) /4
    return avg_pred

def get_binary_prediction(labels, predictions, threshold):
    labels_list = [labels[key] for key in predictions.keys()]
    predictions_list = [predictions[key] for key in predictions.keys()]
    predictions_list_th = [i>threshold for i in predictions_list]
    return labels_list, predictions_list_th

def evaluate_model(predictions, labels):
    """
    Evaluate the accuracy and the AUC of the model.

    Args:
        predictions: Predicted labels.
        threshold: Threshold of the model.
        labels: True labels.
    """
    acc = sum(p == v for p, v in zip(predictions, labels))/len(predictions)
    print("Accuracy :", acc)
    auc_roc = roc_auc_score(labels, predictions)
    print("AUC :", auc_roc)


######################################################################## 
##################### Functions to Save the model ######################
########################################################################

def save_model(model, save_dir, counter, scheduler, train_loss, train_accuracy, val_loss, val_accuracy, best_thresh,
               num_img_3D, b_s, crit, contrast):
    """
    Save the model weights and training information.

    Args:
        model: The neural network model.
        save_dir: Directory to save the model weights and information.
        counter: Current epoch counter.
        scheduler: Learning rate scheduler.
        train_loss: Validation loss.
        train_accuracy: Validation accuracy.
        val_loss: Validation loss.
        val_accuracy: Validation accuracy.
        best_thresh: Best threshold.
        NUM_IMAGES_3D: Number of 3D images.
        BATCH_SIZE: Batch size.
        CRITERION: Loss function.
        CONTRAST: Contrast used.
    """
    os.makedirs(save_dir, exist_ok=True)
    files = os.listdir(save_dir)
    
    if "model_weights.pth" in files:
        os.remove(f"{save_dir}/model_weights.pth")
    torch.save(
        model.state_dict(),
        f"{save_dir}/model_weights.pth",
    )
    
    if "model_summary.txt" in files:
        os.remove(f"{save_dir}/model_summary.txt")
    with open(f"{save_dir}/model_summary.txt", 'w') as f:
        f.write("Model summary:\n")
        f.write("model = ResNet10\n")
        f.write("Parameters :\n")
        f.write(f"   Epochs = {counter}\n   Number_3D_Image = {num_img_3D}\n   Batch_size = {b_s}\n   Learning_rate = {scheduler.get_last_lr()}\n   Optimizer = Adam\n   Loss_function = {crit}\n   Contrast = {contrast}\n")
        f.write(f"\nTrain Results :\n")
        f.write(f"  Train loss: {train_loss}\n  Train accuracy  = {train_accuracy}\n")
        f.write(f"\nValidation Results :\n")
        f.write(f"  Validation loss: {val_loss}\n  Validation accuracy  = {val_accuracy}\n  Threshold = {best_thresh}\n")
        
def save_result(save_dir, loss, accuracy, pred_label):
    with open(f"{save_dir}/model_summary.txt", 'a') as f:
        f.write(f"\nTest Results :\n")
        f.write(f"  Test loss: {loss}\n  Test accuracy  = {accuracy}\n  Prediciton = {pred_label}")

######################################################################## 
###################### Functions to plot metrics #######################
########################################################################
def plot_metrics(results, save_dir_path, show = False):
    # Plot validation and train losses
    plt.figure(figsize=(10, 5))
    plt.plot(results['val_losses'], label=f'Validation Loss')
    plt.plot(results['train_losses'], label=f'Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(f"{save_dir_path}/best_loss_plot.png")
    if show: plt.show()
    
    # Plot validation and train accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(results['val_accuracies'], label=f'Validation accuracy')
    plt.plot(results['train_accuracies'], label=f'Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(f"{save_dir_path}/best_auc_plot.png")
    if show: plt.show()

def plot_all(save_dir, all_results, show = False):
    
    plt.figure(figsize=(20, 10))
 
    # Plot validation accuracies
    plt.subplot(4, 1, 1)
    for c, h in all_results.items():
        plt.plot(h['val_accuracies'], label=f'{c}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot validation losses
    plt.subplot(4, 1, 2)
    for c, h in all_results.items():
        plt.plot(h['val_losses'], label=f'{c}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    # Plot train accuracies
    plt.subplot(4, 1, 3)
    for c, h in all_results.items():
        plt.plot(h['train_accuracies'], label=f'{c}')
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot train losses
    plt.subplot(4, 1, 4)
    for c, h in all_results.items():
        plt.plot(h['train_losses'], label=f'{c}')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_metrics_plot")
    if show: plt.show()
    
######################################################################## 
########################## Helpers functions ###########################
########################################################################
def set_random_seed(seed):
    """
    Set the random seed for PyTorch and NumPy.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def read_csv_to_dict(file_path):
    """
    Read a CSV file and create a dictionary with 'ID' as keys and 'MGMT_value' as values.

    Args:
        file_path: Path to the CSV file.

    Returns:
        dict: Dictionary with 'ID' as keys and 'MGMT_value' as values.
    """
    data_dict = {}

    with open(file_path, 'r') as csv_file:
        # Create a CSV reader
        csv_reader = csv.DictReader(csv_file)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Assuming 'id' and 'MGMT_Value' are column names in your CSV file
            data_dict[int(row['ID'])] = int(row['MGMT_value'])

    return data_dict

def get_mem(i):
    space_res = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
    print(space_res.stdout)
    t = torch.cuda.get_device_properties(0).total_memory/ (1024**2)
    r = torch.cuda.memory_reserved(0)/ (1024**2)
    a = torch.cuda.memory_allocated(0)/ (1024**2)
    f = r-a  # free inside reserved
    print(i, "->", f)

def write_csv(pred_dic, name, save_dir):
    k_s = sorted(pred_dic.keys())
    with open(f'{save_dir}/{name}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['BraTS21ID', 'MGMT_value'])
        for key in k_s:
            writer.writerow([f'{key:05}', pred_dic[key]])