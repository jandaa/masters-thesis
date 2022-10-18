# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
import logging
import numpy as np

log = logging.getLogger(__name__)


def evaluate_scan(pred_ids, gt_ids, confusion, id_to_label_map, ignore_id):

    VALID_CLASS_IDS = list(id_to_label_map.keys())

    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        raise RuntimeError("Ground truth and prediction sizes don't match")

    for (gt_val, pred_val) in zip(gt_ids.flatten(), pred_ids.flatten()):
        if gt_val not in VALID_CLASS_IDS:
            continue
        if pred_val not in VALID_CLASS_IDS:
            pred_val = ignore_id
        confusion[gt_val][pred_val] += 1


def get_iou(label_id, confusion, id_to_label_map):

    VALID_CLASS_IDS = list(id_to_label_map.keys())

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


def write_result_file(confusion, ious, id_to_label_map):

    VALID_CLASS_IDS = list(id_to_label_map.keys())

    log.info("Semantic Segmentation results")
    log.info("iou scores")
    for i in range(len(VALID_CLASS_IDS)):
        label_id = VALID_CLASS_IDS[i]
        label_name = id_to_label_map[label_id]
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
        log.info("{0:<14s}({1:<2d})".format(id_to_label_map[r], VALID_CLASS_IDS[r]))

        output_string = ""
        for c in range(len(VALID_CLASS_IDS)):
            output_string += "\t{0:>5.3f}".format(
                confusion[VALID_CLASS_IDS[r], VALID_CLASS_IDS[c]]
            )
        log.info(output_string)


def evaluate(matches, id_to_label_map, ignore_id, verbose=True):

    VALID_CLASS_IDS = list(id_to_label_map.keys())

    max_id = np.max(VALID_CLASS_IDS)
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    if verbose:
        log.info(f"evaluating {len(matches.keys()) } scans...")

    for scene_name, compare in matches.items():
        evaluate_scan(
            compare["pred"], compare["gt"], confusion, id_to_label_map, ignore_id
        )

    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
        label_id = VALID_CLASS_IDS[i]
        label_name = id_to_label_map[label_id]
        class_ious[label_name] = get_iou(label_id, confusion, id_to_label_map)

    if verbose:
        log.info("classes          IoU")
        log.info("----------------------------")
        for i in range(len(VALID_CLASS_IDS)):
            label_id = VALID_CLASS_IDS[i]
            label_name = id_to_label_map[label_id]
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
        label_id = VALID_CLASS_IDS[i]
        iou_output = get_iou(label_id, confusion, id_to_label_map)
        if type(iou_output) == tuple:
            mean_iou += iou_output[0]
    mean_iou /= len(VALID_CLASS_IDS)

    if verbose:
        log.info("----------------------------")
        log.info(f"mIOU           {mean_iou:.3f}")

    return mean_iou
