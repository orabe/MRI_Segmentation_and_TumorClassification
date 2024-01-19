# source group/bagel/Task_1.B/brainseg_env/bin/activate
# python group/bagel/Task_1.B/main.py

import os
import random
import imageio
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

from sklearn.metrics import classification_report
from tensorflow.keras.utils import plot_model

import tensorflow as tf

from data_handling import *
from model import *
from eval import *
from prediction import *



# plotly
 
# hide tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main():

    """
    # Set GPU device
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate memory on the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    
    
    data_path = '/data/segmentation/train'
    figs_path = '/group/bagel/Task_1.B/MRI_Segmentation_and_TumorClassification/figures'
    evaluation_path = '/group/bagel/Task_1.B/MRI_Segmentation_and_TumorClassification'
    
    segment_classes = {
        0: 'NOT tumor',
        1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
        2: 'EDEMA',
        4: 'ENHANCING'  # original 4 -> converted into 3 later
    }
    
    IMG_SIZE=128
    # [56:184, 56:184, 13:141]
    # Define selected slices range
    VOLUME_START_AT = 0 
    VOLUME_SLICES = 256
    N_CHANNELS = 2 # Number of channels (==2: "T1CE + FLAIR") !! Change vals in __data_generation when modify this param!
    
    
    # 00246
    patient = {'id' : '00006', 'data' : []}
    sample_path = os.path.join(data_path, patient['id'])

    sample = DataLoader(sample_path)
    patient['data'] = sample.explore_sample(patient['id'])
    print(patient['data'][0].shape)

    
#     # Assuming images is a list of 3D numpy arrays
#     # Step 1 & 2: Stack and sum the images
    summed_image = np.sum(np.stack(patient['data'], axis=-1), axis=-1)

#     # Step 3: Find boundaries of non-zero values
    nz = np.nonzero(summed_image)
    min_x, max_x = np.min(nz[0]), np.max(nz[0])
    min_y, max_y = np.min(nz[1]), np.max(nz[1])
    min_z, max_z = np.min(nz[2]), np.max(nz[2])

#     # Step 4: Crop original images using the boundaries
    cropped_images = [img[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] for img in patient['data']]

    print(len(cropped_images))
    print(cropped_images[0].shape)
    print(cropped_images[1].shape)
    print(cropped_images[2].shape)
    print(cropped_images[3].shape)
    print(cropped_images[4].shape)
    
    plot = Plotting(sample_path)
    plot.plot_one_slice_all_mods(cropped_images, patient['id'], slice_nb=64, show_plot=True, save_path=figs_path)
    plot.plot_one_slice_all_mods(patient['data'], patient['id']+'withoutCrop',
                                 slice_nb=64, show_plot=True, save_path=figs_path)
    
    print(np.sum(patient['data'][4][100]!=0), np.sum(patient['data'][4]))
    print(np.sum(patient['data'][4][100]!=0)/ np.sum(patient['data'][4]))
    
    # np.save()

    plot = Plotting(sample_path)
    plot.plot_one_slice_all_mods(patient['data'], patient['id'], slice_nb=100, show_plot=True, save_path=figs_path)
    plot.plot_one_mod_all_slices(patient['data'][0], patient['id'], show_plot=True, save_path=figs_path)
    plot.plot_seg_one_slice(patient['data'][4], patient['id'], show_plot=True, save_path=figs_path)
    plot.plot_segmentation(patient['data'][4][100, :, :], patient['id'], show_plot=True, save_path=figs_path)

    data = DataLoader(data_path)
    
    # this might take some time to run     
    data.explore_seg(segment_classes, '00246')
    
    plot.plot_seg(patient['data'][4][100, :, :], patient['id'], save_path=figs_path)
    
    
    samples_train, samples_val, samples_test = data.split_datset(val_size=0.2, test_size=0.15)

    training_generator = DataGenerator(data_path, samples_train, (IMG_SIZE, IMG_SIZE), 
                                       VOLUME_START_AT, VOLUME_SLICES, N_CHANNELS, batch_size = 1)
    
    valid_generator = DataGenerator(data_path, samples_val, (IMG_SIZE, IMG_SIZE), 
                                    VOLUME_START_AT, VOLUME_SLICES, N_CHANNELS, batch_size = 1)
    
    test_generator = DataGenerator(data_path, samples_test, (IMG_SIZE, IMG_SIZE), 
                                   VOLUME_START_AT, VOLUME_SLICES, N_CHANNELS, batch_size = 1)
    
    # Build the model
    ## Define input data shape
    input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

    # Build and compile the model
    model = build_unet(input_layer, 'he_normal', 0.2)
        
    print(mode.input_shape)
    print(mode.output_shape)
    print(model.summary())

    # plot_model(model, 
    #        show_shapes = True,
    #        show_dtype=False,
    #        show_layer_names = True, 
    #        rankdir = 'TB', 
    #        expand_nested = False, 
    #        dpi = 70)
    
    metrics = ['accuracy',tensorflow.keras.metrics.MeanIoU(num_classes=4),
               dice_coef, precision, sensitivity, specificity, dice_coef_necrotic,
               dice_coef_edema, dice_coef_enhancing]
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics = metrics)

    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.000001, verbose=1),

        keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
                                 verbose=1, save_best_only=True, save_weights_only = True),

        CSVLogger('training.log', separator=',', append=False)
    ]

    # start training
    # model.fit(training_generator,
    #           epochs=30,
    #           steps_per_epoch=len(samples_train),
    #           callbacks=callbacks,
    #           validation_data=valid_generator)
    
    
    model.load_weights("model_.27-0.013691.m5").expect_partial()
    
    # analyze metrics
    plot_acc_loss_iou(show_plot=True, save_path=figs_path)
    # plot_dice(show_plot=True, save_path=figs_path)
    
    # Prediction examples (on testset)
    # Plot Random predictions & Compare with Original (Ground truth)
    
    # todo: refactor other funtions to use this as a param.
    cmap = mpl.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Choose a random patient
    random_sample_id = random.choice(samples_test)
    print(random_sample_id)
    random_sample_path = os.path.join(data_path, random_sample_id)
    
    print('----------;')
    # print(random_sample)
    
    show_predicted_segmentations(model, random_sample_path, random_sample_id, 70, cmap, norm, VOLUME_START_AT, 
                                 IMG_SIZE, VOLUME_SLICES, show_plot=True, save_path=figs_path)
    
    # todo: fix this function
    # showPredictsById(case=samples_train[0][-3:])
    
    # returns original_seg, raw_pred, postpro_pred = 
    original_seg, raw_pred, postpro_pred = show_post_processed_segmentations(model, data_path, random_sample_id, 30,cmap, norm, 
                                                                             VOLUME_START_AT, 
                                                                             VOLUME_SLICES, IMG_SIZE, 
                                                                             show_plot=True, save_path=figs_path)
    
#     print(original_seg.shape)
#     print(raw_pred.shape)
#     print(postpro_pred.shape)
    
#     fig, axstest = plt.subplots(1, 3, figsize=(15, 10))

#     axstest[0].imshow(original_seg, cmap, norm)
#     axstest[0].set_title('Original Segmentation')
    
#     axstest[1].imshow(raw_pred)
#     axstest[1].set_title('Prediction (w/o post processing (layer 1,2,3)')
    
#     axstest[2].imshow(postpro_pred, cmap, norm)
#     axstest[2].set_title('Prediction (w/ post processing (layer 1,2,3)')
    
#     # Add space between subplots
#     plt.subplots_adjust(wspace=0.8)
#     plt.savefig('foo.png')
    
    # Evaluate the model on the test data
#     results = model.evaluate(test_generator, batch_size=100, callbacks= callbacks)

#     print_save_eval(results, evaluation_path)

#     flair_data = os.path.join(data_path, random_sample_id, random_sample_id + '_flair.nii.gz')
#     create_gif(nib.load(flair_data), title=None, filename=os.path.join(figs_path,'flair_MRI.gif'))
    
    # random_sample_id='01322'
    # create_gif(model, data_path, random_sample_id, 'flair', cmap, norm, IMG_SIZE, figs_path)
    
    print(1)
    
   
    
if __name__ == "__main__":
    main()