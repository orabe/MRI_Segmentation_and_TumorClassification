import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.transform import rotate
from skimage.util import montage
import numpy as np
from tqdm import tqdm
import tensorflow
import keras
# import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import Sequence
import cv2


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def explore_sample(self, patient_id):
        """Load the 4 MRI modalities and the segmentation located in the patient's path using the nibabel library
        """
        t1_data = nib.load(os.path.join(self.data_path, patient_id + '_t1.nii.gz')).get_fdata()
        t1ce_data = nib.load(os.path.join(self.data_path, patient_id + '_t1ce.nii.gz')).get_fdata()
        t2_data = nib.load(os.path.join(self.data_path, patient_id + '_t2.nii.gz')).get_fdata()
        flair_data = nib.load(os.path.join(self.data_path, patient_id + '_flair.nii.gz')).get_fdata()
        seg_data = nib.load(os.path.join(self.data_path, patient_id + '_seg.nii.gz')).get_fdata()

        return [t1_data, t1ce_data, t2_data, flair_data, seg_data]
           
        
    def explore_seg(self, segment_classes, sample, sample_idx=0):
        seg_samples = sorted([os.path.join(self.data_path, sample, f"{sample}_seg.nii.gz") 
                              for sample in os.listdir(self.data_path)])
        print(f"Number of segmentation samples: {len(seg_samples)}")
        
        saved_values = []
        max_nb_values = 0

        for sample in tqdm(seg_samples, desc="Loading & analyzing unique segmentations", unit="sample"):
            seg_img = nib.load(sample).get_fdata()
            unique_values = np.unique(seg_img)
            nb_unique_values = len(unique_values)

            if nb_unique_values > max_nb_values:
                max_nb_values = nb_unique_values
                saved_values = unique_values

        print(f"Maximum number of values in all segmentation images: {max_nb_values}")
        print(f"Values: {saved_values}")

        seg_sample = nib.load(seg_samples[sample_idx]).get_fdata()
        values, counts = np.unique(seg_sample, return_counts=True)

        print(f"Distribution of the 4 classes in sample {seg_sample}:")
        for value, count in zip(values, counts):
            class_name = segment_classes.get(value)
            print(f"- Class {value}: {class_name}, Pixels: {count}")

        
    def split_datset(self, val_size=0.2, test_size=0.15):

        # Retrieve all samples from path with listdir().
        # Lists of all files + directories in the specified directory.
        samples = sorted(os.listdir(self.data_path))
        print("Number of all samples:", len (samples))

        # Split the dataset into train and validation sets
        samples_train, samples_val = train_test_split(samples, test_size=val_size, random_state=42)

        # Split the train set into the real train set and in a test set 
        samples_train, samples_test = train_test_split(samples_train, test_size=test_size, random_state=42)

        # Print data distribution
        print('==================================')
        print(f"Trainset size ({int(10*(10-val_size-test_size))}%): {len(samples_train)} samples")
        print(f"Validation size ({int(100*val_size)}%): {len(samples_val)} samples")
        print(f"Testset size ({int(100*test_size)}%): {len(samples_test)} samples")
        print('==================================')
        
        return samples_train, samples_val, samples_test
        
        
        
class Plotting:
    def __init__(self, data_path):
        self.data_path = data_path
    
        
    def plot_one_slice_all_mods(self, data, patient_id, slice_nb, show_plot=True, save_path=None):
        """Plot one slice of the 4 MRI modalities and the segmentation.
        """
        fig, axs = plt.subplots(1, 5, figsize=(15, 15))

        modalities = ['T1', 'T1CE', 'T2', 'FLAIR', 'Segmentation']
        for i, modality in enumerate(modalities):
            axs[i].imshow(data[i][:, :, slice_nb], cmap="gray")
            axs[i].set_title(modality)

        fig.suptitle(f'4 MRI modalities and the segmentation. Patient ID: {patient_id}, Slice Number: {slice_nb}')
        fig.tight_layout()

        if show_plot:
            plt.show()
            

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            fig_filename = f'id{patient_id}_slice{slice_nb}.png'
            fig_path = os.path.join(save_path, fig_filename)
            plt.savefig(fig_path)
            
            print(f"Figure saved at: {fig_path}")
        
        
    def plot_one_mod_all_slices(self, data, patient_id, show_plot=True, save_path=None):
        """Plot all slices of data of one patient, orientation and modality.
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(rotate(montage(data), 90, resize=True), cmap='gray')
        ax.set_title(f'All Slices of Data - Patient ID: {patient_id}')

        if show_plot:
            plt.show()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            fig_filename = f'id{patient_id}_data_slices.png'
            fig_path = os.path.join(save_path, fig_filename)
            plt.savefig(fig_path)
            print(f"Data slices figure saved at: {fig_path}")

            
    def plot_seg_one_slice(self, seg_data, patient_id, show_plot=True, save_path=None):
        """Plot all slices of segmentation of one patient, orientation and modality.
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(rotate(montage(seg_data), 90, resize=True),  cmap="gray")
        ax.set_title(f'All Slices of Segmentation - Patient ID: {patient_id}')

        if show_plot:
            plt.show()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            fig_filename = f'id{patient_id}_seg_slices.png'
            fig_path = os.path.join(save_path, fig_filename)
            plt.savefig(fig_path)
            print(f"Segmentation slices figure saved at: {fig_path}")
    
    
    def plot_segmentation(self, seg_image, patient_id, show_plot=True, save_path=None):
        """Plot a segmentation image.
        """
        cmap = mpl.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])

        norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        plt.imshow(seg_image, cmap=cmap, norm=norm)
        
        plt.colorbar()
        
        if show_plot:
            plt.show()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            fig_filename = f'id{patient_id}_seg_plot.png'
            fig_path = os.path.join(save_path, fig_filename)
            plt.savefig(fig_path)
            print(f"Segmentation plot saved at: {fig_path}")

 
    def plot_seg(self, seg_img, patient_id, show_plot=True, save_path=None):
        
        # Deletion of class 0
        seg_0 = seg_img.copy()
        seg_0[seg_0 != 0] = np.nan

        # Isolation of class 1
        seg_1 = seg_img.copy()
        seg_1[seg_1 != 1] = np.nan

        # Isolation of class 2
        seg_2 = seg_img.copy()
        seg_2[seg_2 != 2] = np.nan

        # Isolation of class 4
        seg_3 = seg_img.copy()
        seg_3[seg_3 != 4] = np.nan

        cmap = mpl.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
        norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


        # Define legend
        class_names = ['class 0', 'class 1', 'class 2', 'class 3']
        legend = [plt.Rectangle((0, 0), 1, 1, color=cmap(i), label=class_names[i]) for i in range(len(class_names))]

        fig, axs3 = plt.subplots(1, 5, figsize=(15, 15))

        axs3[0].imshow(seg_img, cmap=cmap, norm=norm)
        axs3[0].set_title('Original Segmentation')
        axs3[0].legend(handles=legend, loc='upper right')

        axs3[1].imshow(seg_0, cmap=cmap, norm=norm)
        axs3[1].set_title('Not Tumor class 0')

        axs3[2].imshow(seg_1, cmap=cmap, norm=norm)
        axs3[2].set_title('Non-Enhancing Tumor class 1')

        axs3[3].imshow(seg_2, cmap=cmap, norm=norm)
        axs3[3].set_title('Edema class 2')

        axs3[4].imshow(seg_3, cmap=cmap, norm=norm)
        axs3[4].set_title('Enhancing Tumor class 3')

        if show_plot:
            plt.show()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig_filename = f'id{patient_id}classes_seg.png'
            fig_path = os.path.join(save_path, fig_filename)
            plt.savefig(fig_path)
            print(f"Segmented classes plot saved at: {fig_path}")
            
            
#     # show number of data for each dir 
#     def showDataLayout():
#         plt.bar(["Train","Valid","Test"],
#         [len(train_ids), len(val_ids), len(test_ids)], align='center',color=[ 'green','red', 'blue'])
#         plt.legend()

#         plt.ylabel('Number of images')
#         plt.title('Data distribution')

#         plt.show()

   
            

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    """
    def __init__(self, data_path, list_IDs, dim, 
                 volume_start_at=60, volume_slices=75, n_channels = 2, batch_size = 1, shuffle=True):
        """Initialization
        """
        self.data_path = data_path
        self.list_IDs = list_IDs # Patients IDs
        self.dim = dim # Resized image dimensions (128 x 128)
        self.volume_start_at = volume_start_at
        self.volume_slices = volume_slices
        self.n_channels = n_channels # Number of channels (T1CE + FLAIR)
        self.batch_size = batch_size #  Number of images to load each time
        self.shuffle = shuffle # Indicates if data is shuffled for each epoch
        self.on_epoch_end() # Updates indexes after each epoch

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Load & Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        """Generates data containing batch_size samples
        """
        # Initialization
        X = np.zeros((self.batch_size*self.volume_slices, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*self.volume_slices, 240, 240))

        # Generate data
        for c, i in enumerate(Batch_ids):
            
            # Get path of each RMI modality and the segmentation
            sample_path = os.path.join(self.data_path, i, i)
            t1ce_path = sample_path + '_t1ce.nii.gz'
            flair_path = sample_path + '_flair.nii.gz'
            seg_path = sample_path + '_seg.nii.gz'
            #t1_path = sample_path + '_t1.nii.gz'
            #t2_path = sample_path + '_t2.nii.gz'
            
            # Extract the data from these paths
            t1ce = nib.load(t1ce_path).get_fdata()
            flair = nib.load(flair_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()
            #t1 = nib.load(t1_paths).get_fdata()
            #t2 = nib.load(t2_path).get_fdata()
        
            for j in range(self.volume_slices):
                 X[j +self.volume_slices*c,:,:,0] = cv2.resize(flair[:,:,j+self.volume_start_at], self.dim)
                 X[j +self.volume_slices*c,:,:,1] = cv2.resize(t1ce[:,:,j+self.volume_start_at], self.dim)

                 y[j +self.volume_slices*c] = seg[:,:,j+self.volume_start_at]
                    
        # Masks / Segmentations
        y[y==4] = 3
        mask = tensorflow.one_hot(y, 4)
        Y = tensorflow.image.resize(mask, self.dim)
        
        # Scale data between 0 and 1 (since the minimum value in the data is 0)
        return X/np.max(X), Y