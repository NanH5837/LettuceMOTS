import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import lap
from tracking_utils import GIoU

np.random.seed(0)

import global_config as config

def linear_assignment(cost_matrix):
    # _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    # return np.array([[y[i],i] for i in x if i >= 0])
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def matching(detections, storage):

    min_idx = config.min
    max_idx = config.max

    trackers = storage[max(min_idx - 6, 0):min(max_idx + 6, len(storage))]
    # trackers = storage
    cost = np.zeros((len(trackers), len(detections)))
    iou = np.zeros(cost.shape)

    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    for ti, trk in enumerate(trackers):

        for di, det in enumerate(detections):

            issameline = 1 if abs(trk.feature[0] - det[0]) < 100 else 0

            if issameline == 1 and np.abs(det[-1]-trk.feature[-1]) < 0.4:

                df = np.array(det[4:-1]).reshape(1, len(det[4:-1]))
                tf = np.array(trk.feature[4:-1]).reshape(1, len(trk.feature[4:-1]))

                if trk.state == 1: # is not disappear object

                    iou[ti, di] = GIoU(trk.kfpredict[:4], det[:4])
                    cost[ti, di] = cdist(tf, df, metric='cosine')
                    cost[ti, di] = cost[ti, di] * (1 - iou[ti, di])

                else:
                    iou[ti, di] = GIoU(trk.feature[:4], det[:4])
                    cost[ti, di] = cdist(tf, df, metric='cosine')

                cost[ti, di] = cost[ti, di] if (cost[ti, di] < 0.1 and iou[ti, di] > -0.4) else 10000

            else:
                cost[ti, di] = 10000

    matched_indices = linear_assignment(cost)

    matched_indices1 = []

    for m in matched_indices:
        if cost[m[0], m[1]] < 0.1 and iou[m[0], m[1]] > -0.4:
            matched_indices1.append(list(m))
    matched_indices = np.array(matched_indices1)

    if len(storage) > len(trackers) and trackers[0].id != 0:
        for i, mi in enumerate(matched_indices):
            mi[0] = mi[0] + min_idx - 6

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 1]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(storage):
        if (t not in matched_indices[:, 0]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
