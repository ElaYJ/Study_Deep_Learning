import os
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from pycocotools.coco import COCO
from collections import defaultdict

MASK_THRESHOLD = 0.5


def get_class_metrics(id, fname, target_labels, pred_labels):
    # True Positive, False Positive, False Negative 계산
    tp = 0
    fp = 0
    fn = 0
    pred_labels_list = pred_labels.tolist()
    for label in target_labels:
        if label in pred_labels_list:
            tp += 1
            pred_labels_list.remove(label) # 해당 label을 제거하여 중복 counting 방지
        else:
            fn += 1
    fp = len(pred_labels_list)  # 남은 pred_labels는 모두 False Positive

    # Precision, Recall, F1-Score, Accuracy 계산
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = tp / len(target_labels) if len(target_labels) > 0 else 0

    result = {
        'image_id': id,
        'image_fname': fname,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1_score,
        'Accuracy': accuracy
    }
    return result



def combine_masks(masks):
    """
    Combine masks into one image.

    Args:
     @ masks: shape = (N, 1, H, W) or (N, H, W)
    """
    maskimg = np.zeros(masks.shape[-2:])
    for m, mask in enumerate(masks,1):
        mask = mask.squeeze()
        maskimg[mask > MASK_THRESHOLD] = m
    return maskimg

def compute_mask_iou(true_masks, pred_masks):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        true_masks (np array): Labels for ground truth instances.
        pred_masks (np array): predictions

    Returns:
        np array: IoU matrix, of size (true_objects[axis=0]) x (pred_objects[axis=1]).
    """

    true_objects = len(np.unique(true_masks))
    pred_objects = len(np.unique(pred_masks))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        true_masks.flatten(), pred_masks.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.expand_dims(np.histogram(true_masks, bins=true_objects)[0], axis=-1)
    area_pred = np.expand_dims(np.histogram(pred_masks, bins=pred_objects)[0], axis=0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    union[union == 0] = 1e-9
    iou = intersection / union

    return iou

def compute_bbox_iou(gt_bbox, gt_area, pred_bbox):
    l1, t1, r1, b1 = gt_bbox
    l2, t2, r2, b2 = pred_bbox
    pred_area = (r2 - l2) * (b2 - t2)
    intersection = max(0, min(r1, r2) - max(l1, l2)) * max(0, min(b1, b2) - max(t1, t2))
    union = gt_area + pred_area - intersection

    # Compute the IoU
    iou = intersection / union
    return iou

def compute_tp_fp_fn(iou, threshold=0.5):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects of ground true objs
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects of ground true objs
    false_positives = np.sum(matches, axis=0) == 0  # Extra predict objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def get_mask_metrics(id, fname, target_masks, pred_masks):
    true_masks = combine_masks(target_masks)
    pred_masks = combine_masks(pred_masks)
    iou = compute_mask_iou(true_masks, pred_masks)

    all_zero = np.all(iou == 0, axis=1)
    pred_objs_ids = np.argmax(iou, axis=1)
    pred_objs_ids[all_zero] = -1
    pred_objs_iou = np.max(iou, axis=1)

    APs = []
    threshold_rlt = defaultdict()
    for threshold in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = compute_tp_fp_fn(iou, threshold)
        ap = tp / (tp + fp + fn)
        APs.append(ap)
        pc = tp / (tp + fp) if tp + fp > 0 else 0 # precision
        rc = tp / (tp + fn) if tp + fn > 0 else 0 # recall
        f1 = 2 * pc * rc / (pc + rc) if pc + rc > 0 else 0 # f1-score
        ac = tp / len(target_masks) if len(target_masks) > 0 else 0 # accuracy
        # ['TP', 'FP', 'FN', 'AP', 'Precsion', 'Recall', 'F1_score', 'Accuracy']
        threshold_rlt[round(threshold,2)] = [tp, fp, fn, ap, pc, rc, f1, ac]

    mAP = np.mean(APs)
    precision = np.mean([v[4] for k,v in threshold_rlt.items()])
    recall = np.mean([v[5] for k,v in threshold_rlt.items()])
    f1_score = np.mean([v[6] for k,v in threshold_rlt.items()])
    accuracy = np.mean([v[7] for k,v in threshold_rlt.items()])

    result = {
        'image_id': id,
        'image_fname': fname,
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'image_iou': (sum(pred_objs_iou)/len(pred_objs_iou)),
        'objects_iou': pred_objs_iou.tolist()
    }
    mAP70 = [fname] + threshold_rlt.get(0.7)
    # mAP, mAP70, threshold_result, predict_objects_index_order, predict_objects_iou_value
    return result, mAP70, threshold_rlt, pred_objs_ids, pred_objs_iou


def get_bbox_metrics(id, fname, gt_bboxes, gt_areas, pred_bboxes, pred_obj_idx):
    obj_iou = []
    iou = np.zeros((len(gt_bboxes), len(pred_bboxes)))
    for gi, pi in enumerate(pred_obj_idx):
        if pi < 0:
            obj_iou.append(0)
        else:
            gt_bbox = gt_bboxes[gi]
            gt_area = gt_areas[gi]
            pred_bbox = pred_bboxes[pi]
            iou[gi, pi] = compute_bbox_iou(gt_bbox, gt_area, pred_bbox)
            obj_iou.append(iou[gi, pi])

    APs = []
    threshold_rlt = defaultdict()
    for threshold in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = compute_tp_fp_fn(iou, threshold)
        ap = tp / (tp + fp + fn)
        APs.append(ap)
        pc = tp / (tp + fp) if tp + fp > 0 else 0 # precision
        rc = tp / (tp + fn) if tp + fn > 0 else 0 # recall
        f1 = 2 * pc * rc / (pc + rc) if pc + rc > 0 else 0 # f1-score
        ac = tp / len(gt_bboxes) if len(gt_bboxes) > 0 else 0 # accuracy
        threshold_rlt[round(threshold,2)] = [tp, fp, fn, ap, pc, rc, f1, ac]

    mAP = np.mean(APs)
    precision = np.mean([v[4] for k,v in threshold_rlt.items()])
    recall = np.mean([v[5] for k,v in threshold_rlt.items()])
    f1_score = np.mean([v[6] for k,v in threshold_rlt.items()])
    accuracy = np.mean([v[7] for k,v in threshold_rlt.items()])

    result = {
        'image_id': id,
        'image_fname': fname,
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'image_iou': (sum(obj_iou)/len(obj_iou)),
        'objects_iou': obj_iou
    }
    mAP70 = [fname] + threshold_rlt.get(0.7)
    return result, mAP70, threshold_rlt



def get_real_area_from_predict(id, pred_masks):
    obj_info = []
    for i, mask in enumerate(pred_masks):
        mask = mask.squeeze()
        binary_mask = mask > MASK_THRESHOLD
        pixel_area = np.sum(binary_mask)
        real_area_cm2 = pixel_area * 25 * 25
        solar_panel_num = round(real_area_cm2 / (165 * 99))
        obj_info.append({
            'obj_id': i,
            'image_id': id,
            'pixel_area': pixel_area,
            'real_area_cm2': real_area_cm2,
            'solar_panel_num': solar_panel_num
        })
    return obj_info



# targ_masks = target['masks']
# pred_masks = output['masks'].squeeze(axis=1)[keep]
def save_result_image_with_mask(
    save_dir, fname, img, target_masks, predict_masks, iou=0., mAP=0.,
    is_show=False
):
    """
    Params
     @img: 원본 이미지, shape=(3, 512, 512), type=torch.Tensor
     @target_masks: ground truth mask, shape=(N, 512, 512), type=torch.Tensor
     @predict_masks: prediction mask, shape=(N, 512, 512), type=numpy.ndarray
    """
    plt.figure(figsize=(12,4))

    image = img.numpy().transpose((1,2,0))
    plt.subplot(1, 3, 1)
    plt.imshow(image) # permute: 변경하다, 바꾸다; 순서를 바꾸다
    plt.title(f'Origin Image')
    plt.axis('off')

    all_targ_masks = np.zeros(target_masks.shape[-2:])
    for mask in target_masks:
        all_targ_masks = np.logical_or(all_targ_masks, mask)
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(all_targ_masks, alpha=0.3)
    plt.title('Ground Truth')
    plt.axis('off')

    all_pred_masks = np.zeros(predict_masks.shape[-2:])
    for mask in predict_masks:
        all_pred_masks = np.logical_or(all_pred_masks, mask > MASK_THRESHOLD)
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(all_pred_masks, alpha=0.4)
    plt.title(f'Prediction [iou={iou:.2f}, mAP={mAP:.2f}]')
    plt.axis('off')

    save_path = os.path.join(save_dir, f'{fname}_mask_result.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    if is_show:
        plt.tight_layout()
        plt.show()


def convert_to_poly_pairs(seg):
    if isinstance(seg, list):
        return np.array(seg).reshape(int(len(seg)/2), 2)
    else:
        # case_1.
        binary_mask = seg > 0.5
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return [contour.flatten().tolist() for contour in contours[0]]
        # case_2.
        # from skimage import measure
        # contours = measure.find_contours(seg, level=0.5)[0]
        # return [(x, y) for y, x in contours]

def save_result_image_with_bbox_n_mask(
    save_dir, coco_gt, img_id, img, target, predict, categoreis, iou=0., mAP=0.,
    is_show=False
):
    fname = coco_gt.loadImgs(img_id)[0]['file_name'].split(".")[0]
    image = img.numpy().transpose(1, 2, 0)
    _, axs = plt.subplots(1, 2, figsize=(10,5))

    # Ground Truth
    target_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    axs[0].imshow(image)
    for ann in target_anns:
        c = (np.random.random((1, 3))*0.6+0.34).tolist()[0]
        x, y, w, h = ann['bbox']
        axs[0].add_patch(
            Rectangle((x, y), w, h, edgecolor=c, facecolor='none', linewidth=1, linestyle=':')
        )
        # Polygon(xy, closed: bool = True, **kwargs: Any) -> None
        # *xy* is a numpy array with shape Nx2.
        xy_pairs = convert_to_poly_pairs(ann['segmentation'])
        axs[0].add_patch(Polygon(xy_pairs, edgecolor=c, facecolor=c, alpha=0.5))
        axs[0].text(x+2, y+11, categoreis[ann['category_id']], fontsize=10, color='white')
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')


    # Prediction
    axs[1].imshow(image)
    for bbox, mask, label, score in zip(predict['boxes'], predict['masks'], predict['labels'], predict['scores']):
        c = (np.random.random((1, 3))*0.7+0.3).tolist()[0]
        x1, y1, x2, y2 = bbox
        axs[1].add_patch(
            Rectangle((x1, y1), x2-x1, y2-y1, edgecolor=c, facecolor='none', linewidth=1, linestyle=':')
        )
        xy_pairs = convert_to_poly_pairs(mask[0])
        axs[1].add_patch(Polygon(xy_pairs, edgecolor=c, facecolor=c, alpha=0.67))
        axs[1].text(x1, y1-1, f'{categoreis[label]} {score:.2f}', fontsize=10, color='yellow')
    axs[1].set_title(f'Prediction [iou={iou:.2f}, mAP={mAP:.2f}]')
    axs[1].axis('off')

    save_path = os.path.join(save_dir, f'{fname}_bbox_mask_result.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    if is_show:
        plt.tight_layout()
        plt.show()


def test_print():
    print("Import Success~!!")


if __name__ == '__main__':
    print("Import Success~!!")