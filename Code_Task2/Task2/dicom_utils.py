import os
import numpy as np
import pydicom
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ImagesDicomDataset(Dataset):
    def __init__(self, data, 
                 mri_type = "FLAIR", 
                 num_imgs = 6,
                 is_train = True):
        self.data = data
        self.type = mri_type
        self.num_imgs = num_imgs
        self.is_train = is_train
        self.folder = "train" if self.is_train else "test"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        case_id = row.ID
        target = int(row["MGMT_value"])
        _3d_images = self.load_dicom_images_3d(case_id)
        _3d_images = torch.tensor(_3d_images).float()
        if self.is_train:
            return {"id": case_id, "image": _3d_images, "target": target}
        else:
            return {"id": case_id, "image": _3d_images, "case_id": case_id}

    def load_dicom_images_3d(
        self,
        case_id
    ):
        if not (isinstance(case_id, str)):
            case_id = str(case_id).zfill(5)

        path = f"data/classification/{self.folder}/{case_id}/{self.type}"
   
        files = sorted(os.listdir(path), key=lambda x: int(x.split('-')[1].split('.')[0]))
        images = [load_dicom_image(os.path.join(path, f)) for f in files]
        
        if self.type == "FLAIR":
            middle = get_best_center_flair(images) 
            #middle = len(files) // 2
        else:
            middle = len(files) // 2

        num_imgs2 = self.num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        

        
        image_stack = [load_dicom_image(os.path.join(path, f), 256) for f in files[p1:p2+self.num_imgs%2]]
        
        img3d = np.stack(image_stack).T
        if img3d.shape[-1] < self.num_imgs:
            n_zero = np.zeros((256, 256, self.num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d, n_zero), axis=-1)

        if np.min(img3d) < np.max(img3d):
            img3d = img3d - np.min(img3d)
            img3d = img3d / np.max(img3d)

        return np.expand_dims(img3d, 0)

def load_dicom_image(path, img_size=256):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (img_size, img_size))
    return data
    
def get_train_data_generators(csv_path, contrast, 
                        num_imgs = 6, batch_size = 8, 
                        val_size = 0.2, verbose=False):
    ""
    ""
    # Read csv
    csv_file = pd.read_csv(csv_path)
    
    # Split the data
    train_df, val_df = train_test_split(csv_file, test_size = val_size, random_state=42, stratify=csv_file['MGMT_value'])
    train_df = train_df.reset_index(drop=False)
    val_df = val_df.reset_index(drop=False)
    #train_df = csv_file
    
    if verbose:
        print("Taille de l'ensemble d'entraînement :", len(train_df))
        print("Taille de l'ensemble de validation :", len(val_df))
    
    # Create the dataset
    train_dataset = ImagesDicomDataset(data = train_df, 
                                       mri_type = contrast, 
                                       num_imgs = num_imgs)
    val_dataset = ImagesDicomDataset(data=val_df,
                                     mri_type=contrast, 
                                     num_imgs = num_imgs)
    
    # Create dataloader
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=False)
    validation_dl = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=False)
    return train_dl, validation_dl

def get_test_data_generators(csv_path, contrast, 
                        num_imgs = 6, batch_size = 8, 
                        verbose=False):
    ""
    ""
    # Read csv
    csv_file = pd.read_csv(csv_path, sep=';')
    test_df = csv_file
    
    if verbose:
        print("Taille de l'ensemble de test :", len(test_df))
    
    # Create the dataset
    test_dataset = ImagesDicomDataset(data = test_df,
                                      mri_type = contrast,
                                      num_imgs = num_imgs
                                      ,is_train = False)
    
    # Create dataloader
    test_dl = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=False)
    return test_dl

def get_best_center_flair(images):
    
    delta_acc = 0.2
    
    images_array = np.array(images)
    non_zero_values = images_array[images_array != 0]
    if len(non_zero_values) != 0:
        perc_value = np.percentile(non_zero_values, 99.9, axis=None)
        best_center = 0
        count_max = 0
        for i in range(int(len(images_array)*delta_acc), int(len(images_array)*(1-delta_acc))):
            count = np.sum(images_array[i] > perc_value)
            if count > count_max:
                count_max = count
                best_center = i
    else:
        best_center = len(images)//2
    return best_center