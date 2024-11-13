import cv2
import numpy as np
import os


class SpatioTemporal:

    """Generate spatio temporal images from the corpped videos"""

    def __init__(self, end_folder, file_name, pixel_width, ):
        self.end_folder = end_folder
        self.file_name = file_name
        self.pixel_width = pixel_width

    def st_image_load(self, start, height):
        cap = cv2.VideoCapture(self.file_name)
        st_image = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                st_image.append(frame[0:height, start:start+self.pixel_width, :])
            else:
                break
        return st_image

    def st_image(self):
        cap = cv2.VideoCapture(self.file_name)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        limits = [round(width / 6), width - round(width / 6)]
        start = limits[0]
        end = limits[1]
        cnt = 0
        while start <= end:
            original = self.st_image_load(start, height)
            original_reshape = np.hstack(original)
            cv2.imwrite(self.end_folder + os.path.splitext(self.file_name)[0] + '_' + str(cnt) + '.jpg', original_reshape)
            start += self.pixel_width
            cnt += 1