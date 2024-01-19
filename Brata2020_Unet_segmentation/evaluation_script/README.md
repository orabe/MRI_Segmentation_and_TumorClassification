# Evaluation

## Getting started

1. Create new environment or use an existing
2. Install dependencies
    1. ```pip install -r requirements. txt```

## Evaluation

The data folder should have the following structure:

* data/
    * segmentation/
        * test_1/
            * test_1_seg.nii.gz
        * test2/
            * test_2_seg.nii.gz
        * ...
    * classification/
        * submission.csv

### Segmentation

```python evaluate --task segmentation --true-path $TRUE_PATH --pred-path $PRED_PATH```

```$TRUE_PATH``` path to your validation ground-truths

```$PRED_PATH``` path to your validation predictions

### Classification

```python evaluate --task classification --true-path $TRUE_PATH --pred-path $PRED_PATH```

```$TRUE_PATH``` path to your validation ground-truths

```$PRED_PATH``` path to your validation predictions

python evaluate.py  --task segmentation --true-path /group/bagel/Task_1.B/MRI_Segmentation_and_TumorClassification/Brata2020_Unet_segmentation/gt_data_cropped/test/data --pred-path /group/bagel/Task_1.B/MRI_Segmentation_and_TumorClassification/Brata2020_Unet_segmentation/pred_data/test/data