import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_perspective_transform():
    pts_im = np.float32([[227, 170], [342, 154], [269, 327], [444, 290]])
    pts_real = np.float32([[0, 0], [460, 0], [0, 860], [460, 860]])

    return cv2.getPerspectiveTransform(pts_im, pts_real)


def video_capture():

    window_name = 'win'
    mask_name = 'mask'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        return

    M = get_perspective_transform()
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()


    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('m'):
                mask = subtractor.apply(frame)
                detect_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                detect_contours = list(filter(lambda x: cv2.contourArea(x) > frame.shape[0] * frame.shape[1] * (1/500), detect_contours))
                detect_bboxes = list(map(lambda x: cv2.boundingRect(x), detect_contours))
                print(detect_bboxes)
                for x, y, w, h in detect_bboxes:
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (255), 2)
                cv2.imshow(mask_name, mask)




if __name__ == "__main__":
    video_capture()
