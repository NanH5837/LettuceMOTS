import cv2
import numpy as np

def find_contours(mask):

    h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = h[0]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    return contour

def ellpise_feature(contour):
    ellipse = cv2.fitEllipse(contour)
    mi_axis = ellipse[1][0]
    ma_axis = ellipse[1][1]
    axis_ratio = ma_axis / mi_axis
    angle = ellipse[2] / 100
    return axis_ratio, angle

def FD(contour_points):

    Cx, Cy = np.mean(contour_points, axis=0)
    translated_points = contour_points - np.array([Cx, Cy])

    r = np.sqrt(np.sum(np.square(translated_points), axis=1))

    fourier_result = np.fft.fft(r) / len(r)
    fourier_result = np.abs(fourier_result)

    if len(fourier_result) >= 7:
        fourier_result = fourier_result[0:7]
    else:
        a = 7 - len(fourier_result)
        for _ in range(a):
            fourier_result = np.append(fourier_result, 0)

    normalized_fd = fourier_result[2:] / fourier_result[1]

    return normalized_fd

def feature_extraction(masks, dets):
    FE = []
    track_masks = []

    for i in range(len(masks)):
        d = dets[i][:4]
        w = d[2] - d[0]
        h = d[3] - d[1]

        if d[1] > 5 and d[3] < masks.shape[1] - 5 and 150 < d[0] < 600 and w * h > 1000:

            track_masks.append(i)
            m = masks[i]
            local_m = np.array(m[max(int(d[1] - 5), 0):min(int(d[3] + 5), m.shape[0]), min(int(d[0] - 5), 0):max(int(d[2] + 5), m.shape[1])], dtype=np.uint8)

            contour = find_contours(local_m)
            contour = contour[0][:, 0, :]
            nfd = np.array([FD(contour)])
            ellpise_f = np.array(ellpise_feature(contour)).reshape(1, 2)
            f = np.concatenate([[d[:4]], nfd, ellpise_f, np.array([w / h]).reshape(1, 1)], axis=1)
            FE.append(f)

    track_masks = np.array(track_masks)
    FE = np.array(FE)
    FE = FE.reshape(FE.shape[0], FE.shape[2])

    return FE, track_masks