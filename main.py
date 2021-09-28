import cv2
import numpy as np
import matplotlib.pyplot as plt

class BackgroundSubtractor:
    def __init__(self, th=20):
        self.bg = None
        self.th = th

    def set_bg(self, im):
        self.bg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    def apply(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if self.bg is None:
            return np.zeros(gray.shape).astype(np.uint8)
        mask = cv2.absdiff(gray, self.bg)
        mask[mask < self.th] = 0
        mask[mask >= self.th] = 255
        return mask

def get_perspective_transform():
    pts_im = np.float32([[227, 170], [342, 154], [269, 327], [444, 290]])
    pts_real = np.float32([[0, 0], [460, 0], [0, 860], [460, 860]])

    return cv2.getPerspectiveTransform(pts_im, pts_real)

def video_capture():

    window_name = 'win'
    bg_name = 'bg'
    mask_name = 'mask'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(bg_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        return

    M = get_perspective_transform()
    # subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    subtractor = BackgroundSubtractor()
    bg = cv2.imread('./bg.png')
    subtractor.set_bg(bg)
    cv2.imshow(bg_name, bg)

    kernel = np.ones((5,5),np.uint8)

    while True:

        ret, frame = cap.read()

        if ret:
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('b'):
                subtractor.set_bg(frame)
            elif key & 0xFF == ord('m'):
                mask = subtractor.apply(frame)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
                detect_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                # detect_contours = list(filter(lambda x: cv2.contourArea(x) > frame.shape[0] * frame.shape[1] * (1/500), detect_contours))
                detect_bboxes = list(map(lambda x: cv2.boundingRect(x), detect_contours))
                for x, y, w, h in detect_bboxes:
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (255), 2)
                    print(f'bbox {x}, {y}, {w}, {h}')
                    bottom = y + h
                    center = x + w//2
                    pos_im_vec = np.array([[center], [bottom], [1]])
                    pos_real = M.dot(pos_im_vec)
                    pos_real /= pos_real[2]
                    print(f'real_pos x={pos_real[0][0]}mm , y={pos_real[1][0]}mm')
                cv2.imshow(mask_name, mask)




if __name__ == "__main__":
    video_capture()
