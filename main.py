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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from data_handling import *
from model import *
from eval import *
from prediction import *



def create_gif(model, data_path, sample_id, img_type, cmap, norm, IMG_SIZE, figs_path):

    import imageio
    import matplotlib.animation as animate

    img_data = nib.load(os.path.join(data_path, sample_id, sample_id + f'_{img_type}.nii.gz')).get_fdata()
    
    mri_images = []
    original_seg_images = []
    postpro_pred_images = []
    
    fig, axstest = plt.subplots(1, 3, figsize=(15, 10))
    
    for i in range(155):        
        original_seg, _, postpro_pred = show_post_processed_segmentations(model, data_path, sample_id, i,
                                                                                  cmap, norm, VOLUME_START_AT=0, 
                                                                                  VOLUME_SLICES=155,
                                                                                  IMG_SIZE=128, 
                                                                                  show_plot=False)
        
        mri_im = axstest[0].imshow(img_data[..., i], cmap='gray', animated=True)
        mri_images.append([mri_im])
        axstest[0].set_title(f'Raw image({img_type})')
        
        # im_original_seg = axstest[1].imshow(img_data[..., i], cmap='gray', animated=True)
        im_original_seg = axstest[1].imshow(original_seg, cmap, norm)
        original_seg_images.append([im_original_seg])
        axstest[1].set_title('GT Segmentation')

        # im_postpro_pred= axstest[2].imshow(img_data[..., i], cmap='gray', animated=True)
        im_postpro_pred= axstest[2].imshow(postpro_pred, cmap, norm)
        postpro_pred_images.append([im_postpro_pred])
        axstest[2].set_title('Prediction (with post processing)')

        # Add space between subplots
        plt.subplots_adjust(wspace=0.8)
        
        print(i)
    # ani0 = animate.ArtistAnimation(fig, mri_images, interval=1, blit=True, repeat_delay=50)
    ani1 = animate.ArtistAnimation(fig, original_seg_images, interval=1, blit=True, repeat_delay=50)
    # ani2 = animate.ArtistAnimation(fig, postpro_pred_images, interval=1, blit=True, repeat_delay=50)
    
    # plt.title(f'{sample_id}, {img_type}, {IMG_SIZE}x{IMG_SIZE}', fontsize=20)
    # plt.axis('off')
    
    # ani0.save(os.path.join(figs_path, 'raw.gif'))
    ani1.save(os.path.join(figs_path, 'GT.gif'))
    # ani2.save(os.path.join(figs_path, 'prediction.gif'))
    
    # plt.show()
   


   
#     images = []
#     input_image_data = input_image.get_fdata()
#     print(input_image_data.shape)
#     fig = plt.figure()
    
#     for i in range(input_image_data.shape[2]):
#         im = plt.imshow(input_image_data[..., i], cmap='gray', animated=True)
#         images.append([im])
    
#     ani = animate.ArtistAnimation(fig, images, interval=1,\
#         blit=True, repeat_delay=50)
#     plt.title(title, fontsize=20)
#     plt.axis('off')
#     ani.save(filename)
#     plt.show()
    

def main():

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


    data_path = '/data/segmentation/train'
    figs_path = '/group/bagel/Task_1.B/figures'
    evaluation_path = '/group/bagel/Task_1.B/'
    segment_classes = {
        0: 'NOT tumor',
        1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
        2: 'EDEMA',
        4: 'ENHANCING'  # original 4 -> converted into 3 later
    }
    IMG_SIZE=128

    # Define selected slices range
    VOLUME_START_AT = 60 
    VOLUME_SLICES = 75 
    N_CHANNELS = 2 # Number of channels (==2: "T1CE + FLAIR") !! Change vals in __data_generation when modify this param!
    
    
    # 00246
    patient = {'id' : '01322', 'data' : []}
    sample_path = os.path.join(data_path, patient['id'])

    sample = DataLoader(sample_path)
    patient['data'] = sample.explore_sample(patient['id'])

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
    
    show_predicted_segmentations(model, random_sample_path, random_sample_id, 70, cmap, norm, VOLUME_START_AT, IMG_SIZE, VOLUME_SLICES, show_plot=True, save_path=figs_path)
    
    # todo: fix this function
    # showPredictsById(case=samples_train[0][-3:])
    
    # returns original_seg, raw_pred, postpro_pred = 
    original_seg, raw_pred, postpro_pred = show_post_processed_segmentations(model, data_path, random_sample_id, 30,cmap, norm, VOLUME_START_AT, 
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
    
    random_sample_id='01322'
    create_gif(model, data_path, random_sample_id, 'flair', cmap, norm, IMG_SIZE, figs_path)
    
    print(1)
    
    
    
if __name__ == "__main__":
    main()