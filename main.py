import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_perspective_transform():
    pts_im = np.float32([[227, 170], [342, 154], [269, 327], [444, 290]])
    pts_real = np.float32([[0, 0], [460, 0], [0, 860], [460, 860]])

    return cv2.getPerspectiveTransform(pts_im, pts_real)


def video_capture():

    window_name = 'win'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        return

    M = get_perspective_transform()

    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break




if __name__ == "__main__":
    video_capture()
