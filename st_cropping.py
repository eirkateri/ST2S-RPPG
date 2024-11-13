import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import numpy as np
import os
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


class CropVideos:
    """Load video, track face, make sure it can be used for the pyramid, crop video to 10 sec"""

    def __init__(self, final_folder, directory, file_name, pyramid_levels):
        self.file_name = file_name
        self.face_cascade_path = config['cropping']['cascade_path']
        self.pyramid_levels = pyramid_levels
        self.parent_directory = directory
        self.cropped_folder = final_folder
        self.trajectories_path = os.path.join(config['data_paths']['trajectories_path'],
                                              self.file_name.split('.')[0] + '_trajectories.csv')

    def load_video(self, plot, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # UNCOMMENT FOR STABILIZATION
        df = pd.read_csv(self.trajectories_path)

        face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        detected_faces = face_cascade.detectMultiScale(frame)
        if len(detected_faces) == 0:
            return
        f_column, f_row, width, height = detected_faces[0]

        while len(bin(width)[::-1]) - len(bin(width)[::-1].lstrip('0')) < self.pyramid_levels:
            width += 1

        while len(bin(height)[::-1]) - len(bin(height)[::-1].lstrip('0')) < self.pyramid_levels:
            height += 1

        video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')

        # UNCOMMENT FOR STABILIZATION
        x = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                ref_point = [df['x'][x], df['y'][x]]
                video_tensor[x] = frame[int(ref_point[1]) - (height // 2):int(ref_point[1]) - (height // 2) + height,
                                  int(ref_point[0]) - (width // 2):int(ref_point[0]) - (width // 2) + width, :]
                print(x)
                x += 1
            else:
                break

        # CROPPING WITHOUT STABILIZATION
        # x = 0
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if ret is True:
        #         video_tensor[x] = frame[f_row:f_row + height, f_column:f_column + width, :]
        #         x += 1
        #     else:
        #         break
        #     if plot:
        #         cv2.imshow("frame", frame)
        #         k = cv2.waitKey(25) & 0xFF
        #         if k == 27:
        #             break
        return video_tensor, fps

    def save_video(self, fps, video_tensor, i, path):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        [height, width] = video_tensor[0].shape[0:2]
        writer = cv2.VideoWriter(path + os.path.splitext(self.file_name)[0] + '_' + str(i) + ".mp4", fourcc, fps,
                                 (width, height), 1)
        for i in range(0, video_tensor.shape[0]):
            writer.write(cv2.convertScaleAbs(video_tensor[i]))
        writer.release()

    def crop_video(self, plot):
        cap = cv2.VideoCapture(self.parent_directory + self.file_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_length = frame_count / fps
        start = 0
        finish = video_length
        final_video_length = 10
        segments = int(finish // final_video_length)
        for i in range(0, segments):
            ffmpeg_extract_subclip(self.file_name, start, start + final_video_length,
                                   targetname=self.cropped_folder + self.file_name)
            video_tensor, fps = self.load_video(plot, path=self.cropped_folder + self.file_name)
            self.save_video(fps, video_tensor, i, path=self.cropped_folder)
            start += final_video_length
            os.remove(self.cropped_folder + self.file_name)
