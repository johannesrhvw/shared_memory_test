import logging
from typing import TYPE_CHECKING

import numpy as np
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

from camera.frame_correction import FrameCorrection


if TYPE_CHECKING:
    from ids_peak_ipl import ids_peak_ipl


TIMEOUT_LIMIT = 5


class AcquisitionWorker:
    def __init__(self, data_stream: ids_peak.DataStream, frame_processor: FrameCorrection) -> None:
        """
        FrameProvider class for handling the camera
        frame acquisition and processing.

        Args:
            camera (Camera): Source camera for the frame provider.
            frame_processor (FrameProcessor): Image processor
            for color correction of the frame provider.
        """
        self.logger = logging.getLogger(__name__)
        self.frame_processor = frame_processor
        self.data_stream = data_stream
        self.running = False
        self.acquisition_ready = False
        self.frame_number = 0

    def cleanup(self) -> None:
        """
        Cleanup the frame provider.

        Raises:
            ids_peak.InternalErrorException:
            If internal ids_peak exception occurs.
        """
        self.logger.info("Cleaning up frame provider.")
        try:
            self.data_stream.StopAcquisition()
        except ids_peak.InternalErrorException:
            self.logger.exception("ids_peak - InternalException occurred" "while stopping acquisition.")
            raise

    def run(self) -> tuple[np.ndarray, int] | None:
        """
        Run the frame provider.

        Raises:
            ValueError: If the camera settings were not
            locked before calling the function.
        """
        self.running = True
        timeout_counter = 0
        try:
            # Get buffer from device's DataStream. Wait 1000 ms.
            # The buffer is automatically locked until it is queued again.
            frame_number = self.data_stream.NumBuffersDelivered()
            buffer = self.data_stream.WaitForFinishedBuffer(ids_peak.Timeout(5000))
            # Process buffer ...
            ipl_image: ids_peak_ipl.Image = ids_peak_ipl_extension.BufferToImage(buffer)
            image = self.frame_processor.process_image(ipl_image)
            if image is None:
                self.logger.error(f"Frame {self.frame_number} is None.")
                return None
            array: np.ndarray = image.get_numpy()
            self.frame_number += 1
            # Queue buffer so that it can be used again
            self.data_stream.QueueBuffer(buffer)
        except ids_peak.TimeoutException:
            timeout_counter += 1
            if timeout_counter >= TIMEOUT_LIMIT:
                self.logger.exception(f"Timeout limit of{TIMEOUT_LIMIT} reached.")
                raise
            self.logger.exception("ids_peak - TimeoutException occurred " "while waiting for buffer.")
            return None
        except Exception:
            self.logger.exception("ids_peak - InternalException occurred " "while waiting and requeueing for buffer.")
            return None
        return array, frame_number
