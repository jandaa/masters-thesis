# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
import logging
import os, sys
import numpy as np
import math

# import util.utils_3d as util_3d
# import util.utils as util

log = logging.getLogger(__name__)

CLASS_LABELS = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
VALID_CLASS_IDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)
UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1


def evaluate_scan(pred_ids, gt_ids, confusion):
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        raise RuntimeError("Ground truth and prediction sizes don't match")

    for (gt_val, pred_val) in zip(gt_ids.flatten(), pred_ids.flatten()):
        if gt_val not in VALID_CLASS_IDS:
            continue
        if pred_val not in VALID_CLASS_IDS:
            pred_val = UNKNOWN_ID
        confusion[gt_val][pred_val] += 1


def get_iou(label_id, confusion):
    if not label_id in VALID_CLASS_IDS:
        return float("nan")
    # #true positives
    tp = np.longlong(confusion[label_id, label_id])
    # #false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # #false positives
    not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return (float(tp) / denom, tp, denom)


def write_result_file(confusion, ious):
    log.info("Semantic Segmentation results")
    log.info("iou scores")
    for i in range(len(VALID_CLASS_IDS)):
        label_id = VALID_CLASS_IDS[i]
        label_name = CLASS_LABELS[i]
        if type(ious[label_name]) == tuple:
            iou = ious[label_name][0]
            log.info("{0:<14s}({1:<2d}): {2:>5.3f}".format(label_name, label_id, iou))
    log.info("confusion matrix")
    log.info("\t\t\t")

    output_string = ""
    for i in range(len(VALID_CLASS_IDS)):
        # f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
        output_string += "{0:<8d}".format(VALID_CLASS_IDS[i])
    log.info(output_string)

    for r in range(len(VALID_CLASS_IDS)):
        log.info("{0:<14s}({1:<2d})".format(CLASS_LABELS[r], VALID_CLASS_IDS[r]))

        output_string = ""
        for c in range(len(VALID_CLASS_IDS)):
            output_string += "\t{0:>5.3f}".format(
                confusion[VALID_CLASS_IDS[r], VALID_CLASS_IDS[c]]
            )
        log.info(output_string)


def evaluate(matches, verbose=True):
    max_id = UNKNOWN_ID
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    if verbose:
        log.info(f"evaluating {len(matches.keys()) } scans...")

    for scene_name, compare in matches.items():
        evaluate_scan(compare["pred"], compare["gt"], confusion)

    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_ious[label_name] = get_iou(label_id, confusion)

    if verbose:
        log.info("classes          IoU")
        log.info("----------------------------")
        for i in range(len(VALID_CLASS_IDS)):
            label_name = CLASS_LABELS[i]
            if type(class_ious[label_name]) == tuple:
                log.info(
                    "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                    )
                )

    # Return mean IOU
    mean_iou = 0
    for i in range(len(VALID_CLASS_IDS)):
        mean_iou += get_iou(label_id, confusion)
    
    return mean_iou / len(VALID_CLASS_IDS)
