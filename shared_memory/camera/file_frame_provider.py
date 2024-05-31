import logging
import os

import cv2
import numpy as np
from configuration.config import CameraConfig


TIMEOUT_LIMIT = 5


class FileFrameProvider:
    def __init__(self, source_file: str, config: CameraConfig) -> None:
        """
        Frame provider class to provide frames from a video file.
        Used for testing without connected Cameras.

        Args:
            source_file (str): video file path.
            config (CameraConfig): config for the frame provider containing
            desired width, height and offset.
        """
        self.logger = logging.getLogger(__name__)
        if os.path.isfile(source_file):
            self.source_file = source_file
        else:
            raise ValueError(f"File {source_file} does not exist.")
        self.running = False
        self.acquisition_ready = False
        settings = config.device_settings
        self.image_width = settings.roi_width
        self.image_height = settings.roi_height
        self.offset_x = settings.x_offset
        self.offset_y = settings.y_offset
        self.data_stream = cv2.VideoCapture(self.source_file)
        while not self.data_stream.isOpened():
            self.data_stream = cv2.VideoCapture(self.source_file)
            cv2.waitKey(1000)

    def cleanup(self) -> None:
        """
        Cleanup the frame provider.
        """
        self.data_stream.release()
        self.logger.info("FileFrameProvider cleanup.")

    def run(self) -> tuple[np.ndarray, int] | None:
        """
        Run the frame provider.

        Returns:
            tuple[np.ndarray, int]: Next frame from the source
            file and the corresponding frame number.
        """

        pos_frame = int(self.data_stream.get(cv2.CAP_PROP_POS_FRAMES))
        flag, frame = self.data_stream.read()
        if flag:
            # frame ready and captured
            resized_frame = frame[
                self.offset_y : self.offset_y + self.image_height,
                self.offset_x : self.offset_x + self.image_width,
            ]
            pos_frame = int(self.data_stream.get(cv2.CAP_PROP_POS_FRAMES))
            return resized_frame, pos_frame

        if self.data_stream.get(cv2.CAP_PROP_POS_FRAMES) == self.data_stream.get(cv2.CAP_PROP_FRAME_COUNT):
            # number of captured frames equal to total number of frames, stop
            self.logger.info("End of video file.")
            self.data_stream.release()
            return None
        # frame not ready
        self.logger.debug(f"Frame {pos_frame} is not ready.")
        self.data_stream.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
        # wait for next frame
        cv2.waitKey(100)
        return None
