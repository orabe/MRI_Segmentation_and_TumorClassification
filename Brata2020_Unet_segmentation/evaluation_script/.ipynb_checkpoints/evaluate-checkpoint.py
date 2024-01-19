import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from argparse import ArgumentParser

from sklearn.metrics import f1_score, fbeta_score, accuracy_score
from sklearn.metrics import roc_auc_score as auc
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose, AsDiscrete


def reformat_multiclass_for_monai(image):
    """
    Prepare the image data as a tensor for monai
    :param image:
    :return:
    """

    # add the first dimension for the channel
    data_c = np.expand_dims(image, 0)

    # change class 4 into 3 to have consecutive classes
    data_c[data_c == 4] = 3

    # transform to one-hot format with 4 classes and add batch dimension
    make_4_class = Compose([AsDiscrete(to_onehot=4), lambda x: x[None]])

    data_out = make_4_class(data_c)
    return data_out


def reformat_binary_for_monai(image):
    """
    Prepare the image data as a tensor for monai
    :param image:
    :return:
    """
    # add the first dimension for the channel
    data_c = np.expand_dims(image, 0)

    # transform to one-hot format with 4 classes and add batch dimension
    make_binary = Compose([AsDiscrete(), lambda x: x[None]])
    data_out = make_binary(data_c)
    return data_out


def evaluate_segmentation(path_true, path_pred):
    """
    Evaluate the segmentation on DICE and Hausdorff distance
    :param path_true: path to the ground-truth images
    :param path_pred: path to the predicted images
    :return:
    """

    # define metrics
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False, ignore_empty=False)
    hd = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", get_not_nans=False)
    cn = ['ID', 'dice_1', 'dice_2', 'dice_4', 'hd_tc', 'hd_wt']
    df = pd.DataFrame(columns=cn)

    # iterate over all images
    images = path_true.glob('*')
    images = list(filter(lambda image: image.is_dir(), images))
    images = sorted(images, key=lambda x: int(x.parts[-1].split('_')[-1]))
    i = 0
    for p in images:
        print(i)
        i+=1
        #
        case = p.parts[-1]
        case_metrics = [case]

        # load images
        img_true = nib.load(path_true / case / f"{case}_seg.nii.gz")
        img_pred = nib.load(path_pred / case / f"{case}_seg.nii.gz")

        data_true = img_true.get_fdata().astype(int)
        data_pred = img_pred.get_fdata().astype(int)
        data_true_t = reformat_multiclass_for_monai(data_true)
        data_pred_t = reformat_multiclass_for_monai(data_pred)

        metric = dice_metric(data_pred_t, data_true_t).numpy().tolist()
        case_metrics.append(metric[0][0])
        case_metrics.append(metric[0][1])
        case_metrics.append(metric[0][2])

        # get tumor core
        tc_true = np.isin(data_true, [1, 4]).astype(int)
        tc_pred = np.isin(data_pred, [1, 4]).astype(int)
        tc_true_t = reformat_binary_for_monai(tc_true)
        tc_pred_t = reformat_binary_for_monai(tc_pred)
        h_tc_metric = hd(y_pred=tc_pred_t, y=tc_true_t).numpy().tolist()
        h_tc_metric = h_tc_metric[0][0]
        case_metrics.append(h_tc_metric)

        # get tumor
        wt_true = np.isin(data_true, [1, 2, 4]).astype(int)
        wt_pred = np.isin(data_pred, [1, 2, 4]).astype(int)
        wt_true_t = reformat_binary_for_monai(wt_true)
        wt_pred_t = reformat_binary_for_monai(wt_pred)

        # calculate hausdorff distances
        h_wt_metric = hd(y_pred=wt_pred_t, y=wt_true_t).numpy().tolist()
        hd.reset()
        h_wt_metric = h_wt_metric[0][0]
        case_metrics.append(h_wt_metric)

        # reset metrics
        dice_metric.reset()
        hd.reset()
        df = pd.concat([df, pd.DataFrame([case_metrics], columns=cn)], ignore_index=True)

    return df


def evaluate_classification(path_true, path_pred):
    """
    Evaluate the classification on F1-score and AUC
    :param path_true: path to ground-truth CSV file
    :param path_pred: path to predictions CSV file
    :return:
    """

    # load data
    true_labels = pd.read_csv(path_true)
    pred_labels = pd.read_csv(path_pred)

    # merge labels
    # TODO: check cases where labels are missing or assigned to unknown cases
    merged_df = true_labels.merge(pred_labels, suffixes=("_t", "_p"), on='ID')

    # calculate metrics
    metrics_df = [{
        'F1-score': f1_score(merged_df['MGMT_value_t'], merged_df['MGMT_value_p']),
        'F2-score': fbeta_score(merged_df['MGMT_value_t'], merged_df['MGMT_value_p'], beta=2.),
        'Accuracy': accuracy_score(merged_df['MGMT_value_t'], merged_df['MGMT_value_p']),
        'AUC': auc(merged_df['MGMT_value_t'], merged_df['MGMT_probability'])
    }]

    # create dataframe
    metrics_df = pd.DataFrame(metrics_df)
    return metrics_df


def main():
    """

    :return:
    """

    # add arguments
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="segmentation", choices=["classification", "segmentation"])
    parser.add_argument("--true-path", type=Path, default='data/test')
    parser.add_argument("--pred-path", type=Path, default='data/group')
    args = parser.parse_args()

    #
    if args.task == "segmentation":
        metrics_df = evaluate_segmentation(args.true_path / 'segmentation', args.pred_path / 'segmentation')
        metrics_df.to_csv(args.pred_path / 'segmentation-metrics.csv', index=False)
    elif args.task == "classification":
        metrics_df = evaluate_classification(
            args.true_path / 'classification' / 'labels.csv', args.pred_path / 'classification' / 'submission.csv'
        )
        metrics_df.to_csv(args.pred_path / 'classification-metrics.csv', index=False)
    else:
        raise ValueError(f'{args.task} is an unknown task')


if __name__ == "__main__":
    main()
