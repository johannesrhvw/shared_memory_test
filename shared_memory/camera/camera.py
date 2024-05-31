import functools
import logging
from collections.abc import Callable
from typing import Any
from typing import TypeVar
from typing import cast

from configuration.config import CameraConfig
from ids_peak import ids_peak


class NoDatastreamError(Exception):
    pass


class NoNodeMapError(Exception):
    pass


class LockSettingsError(Exception):
    pass


class AcquisitionError(Exception):
    pass


FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def check_camera_status(func: FuncT) -> FuncT:
    @functools.wraps(func)
    def wrapper(self: "Camera", *args: Any, **kwargs: Any) -> Any | None:
        if self.settings_locked or self.running:
            logging.warning(f"{func.__name__} failed for {self.serial_number}: " "locked settings or running camera.")
            return None
        return func(self, *args, **kwargs)

    return cast(FuncT, wrapper)


class Camera:
    def __init__(self, device: ids_peak.DeviceDescriptor) -> None:
        """
        Initializes the Camera with the given device and process queue.

        Args:
            name (str): user_id to be set for the camera.
            device (ids_peak.Device): camera device descriptor.

        Raises:
            NoDatastreamError: Raised if the device
            has no available datastreams.
        """
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.settings_locked = False
        self.device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
        # Get remote device from device and corresponding node map
        self.remote_device: ids_peak.RemoteDevice = self.device.RemoteDevice()
        node_maps: ids_peak.VectorNodeMap = self.remote_device.NodeMaps()
        self.node_map_remote: ids_peak.NodeMap = node_maps[0]
        # Get datastreams from device descriptor and assign
        # its remote data stream to the cams worker
        self.data_stream = self._get_datastream()
        self._set_bufferhandling_newest_only()
        self.serial_number = self.node_map_remote.FindNode("DeviceSerialNumber").Value()
        self.logger.debug(f"Opened camera with serial number: {self.serial_number}.")

    def _lock_settings(self) -> None:
        """
        Locks the cameras settings to prevent changes during acquisition.

        Returns:
            bool: True if the settings were locked, False otherwise.
        """
        if self.settings_locked is True:
            self.logger.info("Settings are already locked.")
            return
        access = self.node_map_remote.FindNode("TLParamsLocked").AccessStatus()
        if access in {
            ids_peak.NodeAccessStatus_ReadWrite,
            ids_peak.NodeAccessStatus_WriteOnly,
        }:
            self.node_map_remote.FindNode("TLParamsLocked").SetValue(1)
            self.settings_locked = True
            return
        self.logger.error("Failed to lock settings.")
        raise LockSettingsError("Failed to lock settings. Bad AccessStatus or NodeType.")

    def _unlock_settings(self) -> None:
        """
        Unlocks the cameras settings to allow changes.

        Returns:
            bool: True if the settings were unlocked, False otherwise.
        """
        if self.settings_locked is False:
            self.logger.info("Settings are already unlocked.")
            return
        access = self.node_map_remote.FindNode("TLParamsLocked").AccessStatus()
        if access in {
            ids_peak.NodeAccessStatus_ReadWrite,
            ids_peak.NodeAccessStatus_WriteOnly,
        }:
            self.node_map_remote.FindNode("TLParamsLocked").SetValue(False)
            self.settings_locked = False
            return
        self.logger.error("Failed to unlock settings.")
        raise LockSettingsError("Failed to unlock settings. Bad AccessStatus or NodeType.")

    def _set_bufferhandling_newest_only(self) -> None:
        node_map_datastream = self.data_stream.NodeMaps()[0]
        current_status = node_map_datastream.FindNode("StreamBufferHandlingMode").CurrentEntry().SymbolicValue()
        self.logger.debug("Current StreamBufferHandlingMode: " f"{current_status}.")
        all_entries = node_map_datastream.FindNode("StreamBufferHandlingMode").Entries()
        available_entries = [
            entry.SymbolicValue()
            for entry in all_entries
            if (
                entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented
            )
        ]

        # Set StreamBufferHandlingMode to "OldestFirst" (str)
        if "NewestOnly" in available_entries:
            node_map_datastream.FindNode("StreamBufferHandlingMode").SetCurrentEntry("NewestOnly")

    def _set_start_acquisition(self) -> None:
        """
        Starts the image acquisition process.

        This function sets the payload size for the acquisition
        worker for buffer allocation.
        It also locks critical camera features to prevent them
        from changing during acquisition.
        Starts Camera Acquisition, the Cameras DataStream via
        AcquisitionWorker and the AcquisitionWorker thread.

        Raises:
            AcquisitionError: If camera is already rnning or the settings
            were not locked befor starting the camera.
        """
        if not self.running:
            self.node_map_remote.FindNode("AcquisitionStart").Execute()
            self.node_map_remote.FindNode("AcquisitionStart").WaitUntilDone()
            self.running = True
            return
        self.logger.error("Failed to start acquisition. Camera is already " "running or settings are not locked.")
        raise AcquisitionError("Failed to start acquisition. Camera is already" "running or settings are not locked.")

    def _set_stop_acquisition(self) -> None:
        """
        Halts the image acquisition process.

        Raises:
            AcquisitionError: If the camera is not running or the settings
            were not locked.
        """
        if self.running and self.settings_locked:
            # self.acquisition_worker.stop_acquisition()
            self.node_map_remote.FindNode("AcquisitionStop").Execute()
            # self.acquisition_worker.stop()
            self.running = False
            return
        self.logger.error("Failed to stop acquisition. Camera is " "not running or settings are not locked.")
        raise AcquisitionError("Failed to stop acquisition. Camera is " "not running or settings are not locked.")

    def _flush_and_revoke_buffers(self) -> None:
        """
        Flush and revoke all buffers from the DataStream.
        """
        try:
            self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in self.data_stream.AnnouncedBuffers():
                self.data_stream.RevokeBuffer(buffer)
        except ids_peak.InternalErrorException:
            self.logger.exception("ids_peak - InternalException occurred while " "flushing and revoking buffers.")
            raise
        except ids_peak.InvalidArgumentException:
            self.logger.exception("ids_peak - InvalidArgumentException " "occurred while revoking buffers.")
            raise

    def _allocate_buffers(self, payload_size: int, num_buffers: int) -> None:
        """
        Allocate and announce buffers with the
        given payload size and number of buffers.

        Args:
            payload_size (int): Size of the payload.
            num_buffers (int): Number of buffers to allocate with payload size.
        """
        try:
            for _ in range(num_buffers):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)
        except ids_peak.InternalErrorException:
            self.logger.exception("ids_peak - InternalException allocanounce " "and queue buffers.")
            raise
        except ids_peak.InvalidArgumentException:
            self.logger.exception(
                "ids_peak - queueing buffers payload size is " "outside the valid range " "or is not supported."
            )
            raise
        except ids_peak.BadAllocException:
            self.logger.exception("ids_peak - bad memory allocation of buffers.")
            raise

    def _create_buffer(self, payload_size: int) -> None:
        """
        Create a buffer with the given payload size.

        Args:
            payload_size (int): The size of the payload.
        """
        if self.data_stream:
            self._flush_and_revoke_buffers()
            try:
                num_buffers_min_required = self.data_stream.NumBuffersAnnouncedMinRequired()
            except ids_peak.InternalErrorException:
                self.logger.exception("ids_peak - InternalException while " "getting number of minimum buffers.")
                self._flush_and_revoke_buffers()
                raise
            else:
                self.logger.info(
                    f"Minimum number of " f"{num_buffers_min_required} " f"buffers with size {payload_size}."
                )
                self._allocate_buffers(payload_size, num_buffers_min_required)

    def apply_settings(self, config: CameraConfig) -> None:
        settings = config.device_settings
        self.set_roi(
            settings.x_offset,
            settings.y_offset,
            settings.roi_width,
            settings.roi_height,
        )
        self.set_exposure(settings.exposure)
        self.set_frame_rate(settings.fps)

    def prepare_acquisition_start(self) -> ids_peak.DataStream:
        payload = self.get_payload_size()
        if payload is None:
            self.logger.info(f"Failed to get payload size " f"for camera {self.serial_number}")
            raise
        # self._lock_settings()
        self._create_buffer(payload)
        self.data_stream.StartAcquisition(
            ids_peak.AcquisitionStartMode_Default,
            ids_peak.DataStream.INFINITE_NUMBER,
        )
        self._set_start_acquisition()
        return self.data_stream

    def cleanup_acquisition_stop(self) -> None:
        self.data_stream.StopAcquisition()
        self._set_stop_acquisition()
        self._flush_and_revoke_buffers()
        self._unlock_settings()

    def set_file(self, file: str) -> None:
        self.node_map_remote.LoadFromFile(file)

    @check_camera_status
    def get_id(self) -> str:
        """
        Get the Cameras ID.

        Returns:
            str: ID in string format.
        """
        cam_id: str = self.device.ID()
        return cam_id

    @check_camera_status
    def get_display_name(self) -> str:
        """
        Get the Cameras Display Name.

        Returns:
            str: Display Name in string format.
        """
        cam_display_name: str = self.device.DisplayName()
        return cam_display_name

    @check_camera_status
    def get_device_userid(self) -> str:
        """
        Get the Cameras User ID.

        Returns:
            str | None: User ID in string format.
        """
        return self.node_map_remote.FindNode("DeviceUserID").Value()

    @check_camera_status
    def get_access_status(self) -> str:
        """
        Get the Cameras Access Status.

        Returns:
            str: Access Status in string format.
        """
        cam_access: str = self.device.AccessStatus()
        return cam_access

    @check_camera_status
    def get_information(self) -> dict:
        """
        Retrieves information about the camera.

        Returns:
            dict: dict containing the cameras info in the keys ID, Serial,
            DisplayName, Model, AccessStatus.
        """
        cam_id = self.device.ID()
        cam_serial = self.device.SerialNumber()
        cam_model = self.device.ModelName()
        cam_access = self.device.AccessStatus()
        cam_display_name = self.device.DisplayName()
        return {
            "ID": cam_id,
            "Serial": cam_serial,
            "DisplayName": cam_display_name,
            "Model": cam_model,
            "AccessStatus": cam_access,
        }

    @check_camera_status
    def _get_datastream(self) -> ids_peak.DataStream:
        """
        Retrieves the camera's datastream.

        Returns:
            ids_peak.DataStream: The camera's datastream.
        Raises:
            NoDatastreamError: If the device does not contain
            any datastreamdescriptor.
        """
        datastreamdescriptors: list[ids_peak.DataStreamDescriptor] = self.device.DataStreams()
        if len(datastreamdescriptors) == 0:
            self.logger.error("No datastreams found.")
            raise NoDatastreamError("Device node does not contain any datastreamdescriptor.")
        return datastreamdescriptors[0].OpenDataStream()

    @check_camera_status
    def get_payload_size(self) -> int | None:
        """
        Retrieves the payload size of the camera.

        Returns:
            int | None: The payload size of the camera.
        """
        return self.node_map_remote.FindNode("PayloadSize").Value()

    @check_camera_status
    def get_frame_rate(self) -> float | None:
        """
        Retrieves the frame rate of the camera.

        Returns:
            float | None: The frame rate of the camera.
        """
        return self.node_map_remote.FindNode("AcquisitionFrameRate").Value()

    @check_camera_status
    def get_exposure(self) -> float | None:
        """
        Retrieves the exposure time of the camera.

        Returns:
            float | None: The exposure time of the camera.
        """
        return self.node_map_remote.FindNode("ExposureTime").Value()

    @check_camera_status
    def get_frame_size(self) -> tuple[int, int] | None:
        """
        Retrieves the frame size of the camera.

        Returns:
            tuple[int, int] | None: tuple with width and height of the frame.
        """
        width = self.node_map_remote.FindNode("Width").Value()
        height = self.node_map_remote.FindNode("Height").Value()
        return width, height

    @check_camera_status
    def get_roi(self) -> tuple[int, int, int, int] | None:
        """
        Retrieves the Region Of Interest (ROI) of the camera.

        Returns:
            tuple[int, int, int, int] | None: tuple with x, y,
            width and height of the ROI.
        """
        x = self.node_map_remote.FindNode("OffsetX").Value()
        y = self.node_map_remote.FindNode("OffsetY").Value()
        width = self.node_map_remote.FindNode("Width").Value()
        height = self.node_map_remote.FindNode("Height").Value()
        return x, y, width, height

    @check_camera_status
    def get_pixelformat(self) -> tuple[str, list[str]] | None:
        """
        Retrieves the pixel format of the camera.

        Returns:
            tuple[str, list[str]] | None: tuple containing the current format
            and a list of available formats.
        """
        # Determine the current entry of PixelFormat (str)
        value = self.node_map_remote.FindNode("PixelFormat").CurrentEntry().SymbolicValue()
        # Get a list of all available entries
        # of PixelFormat using list comprehension
        available_entries = [
            entry.SymbolicValue()
            for entry in self.node_map_remote.FindNode("PixelFormat").Entries()
            if entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
            and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented
        ]
        return value, available_entries

    @check_camera_status
    def get_maximum_throughput(self) -> int | None:
        """
        Gets the maximum throughput of the camera.

        Returns:
            int | None: The maximum throughput in MBit/s.
        """
        return self.node_map_remote.FindNode("DeviceLinkThroughputLimit").Maximum()

    @check_camera_status
    def set_throughput(self, throughput: int) -> int | None:
        """
        Sets the throughput of the camera.

        Args:
            throughput (int): the desired throughput in MBit/s.
        Returns:
            int | None: The set throughput in MBit/s.
        """
        inc: int = self.node_map_remote.FindNode("DeviceLinkThroughputLimit").Increment()
        throughput = min(
            max(
                throughput,
                (self.node_map_remote.FindNode("DeviceLinkThroughputLimit").Minimum()),
            ),
            (self.node_map_remote.FindNode("DeviceLinkThroughputLimit").Maximum()),
        )
        throughput = throughput - (throughput % inc)
        self.node_map_remote.FindNode("DeviceLinkThroughputLimit").SetValue(throughput)
        return throughput

    @check_camera_status
    def set_exposure(self, exposure: float) -> float | None:
        """
        Set the exposure time of the camera.

        Args:
            exposure (float): the desired exposure time.

        Returns:
            float | None: the set exposure time.
        """
        min_exposure_time = 0
        max_exposure_time = 0
        inc_exposure_time = 0

        # Get exposure range. All values in microseconds
        min_exposure_time = self.node_map_remote.FindNode("ExposureTime").Minimum()
        max_exposure_time = self.node_map_remote.FindNode("ExposureTime").Maximum()

        if self.node_map_remote.FindNode("ExposureTime").HasConstantIncrement():
            inc_exposure_time = self.node_map_remote.FindNode("ExposureTime").Increment()
        else:
            # If there is no increment, it might be useful to choose a
            # suitable increment for a GUI control element
            inc_exposure_time = 1000

        exposure = exposure - (exposure % inc_exposure_time)
        exposure = min(max(exposure, min_exposure_time), max_exposure_time)

        # Set exposure time to minimum
        self.node_map_remote.FindNode("ExposureTime").SetValue(exposure)
        return exposure

    @check_camera_status
    def set_frame_rate(self, frame_rate: float) -> float | None:
        """
        Set the frame rate of the camera.

        Args:
            frame_rate (float): the desired frame rate.

        Returns:
            float | None: the set frame rate.
        """
        # Get frame rate range and set frame rate
        min_frame_rate = self.node_map_remote.FindNode("AcquisitionFrameRate").Minimum()
        max_frame_rate = self.node_map_remote.FindNode("AcquisitionFrameRate").Maximum()
        frame_rate = max(min(frame_rate, max_frame_rate), min_frame_rate)
        self.node_map_remote.FindNode("AcquisitionFrameRate").SetValue(frame_rate)
        return frame_rate

    @check_camera_status
    def set_roi(self, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        """
        Set the Region Of Interest (ROI) of the camera.

        Args:
            x (int): offset x of roi
            y (int): offset y of roi
            width (int): width of roi
            height (int): height of roi

        Returns:
            tuple[int, int, int, int] | None: the set ROI as a tuple of x, y,
            width and height.
        """
        # Get the minimum ROI
        x_min = self.node_map_remote.FindNode("OffsetX").Minimum()
        y_min = self.node_map_remote.FindNode("OffsetY").Minimum()
        w_min = self.node_map_remote.FindNode("Width").Minimum()
        h_min = self.node_map_remote.FindNode("Height").Minimum()

        # Set the minimum ROI. This removes any size restrictions
        # due to previous ROI settings
        self.node_map_remote.FindNode("OffsetX").SetValue(x_min)
        self.node_map_remote.FindNode("OffsetY").SetValue(y_min)
        self.node_map_remote.FindNode("Width").SetValue(w_min)
        self.node_map_remote.FindNode("Height").SetValue(h_min)

        # Get the maximum ROI values
        x_max = self.node_map_remote.FindNode("OffsetX").Maximum()
        y_max = self.node_map_remote.FindNode("OffsetY").Maximum()
        w_max = self.node_map_remote.FindNode("Width").Maximum()
        h_max = self.node_map_remote.FindNode("Height").Maximum()

        # Get the increment
        x_inc = self.node_map_remote.FindNode("OffsetX").Increment()
        y_inc = self.node_map_remote.FindNode("OffsetY").Increment()
        w_inc = self.node_map_remote.FindNode("Width").Increment()
        h_inc = self.node_map_remote.FindNode("Height").Increment()

        # Check that the ROI parameters are within their valid range

        if x > x_max:
            x = x_max
        if y > y_max:
            y = y_max
        if y < y_min:
            y = y_min
        if x < x_min:
            x = x_min

        total_width = x + width
        total_height = y + height

        if total_width > w_max:
            overshoot = total_width - w_max
            width = width - overshoot
        if total_height > h_max:
            overshoot = total_height - h_max
            height = height - overshoot
        # make sure that difference between minimum and desired
        # values are multiples of the increment
        x = x - ((x - x_min) % x_inc)
        y = y - ((y - y_min) % y_inc)
        width = width - ((width - w_min) % w_inc)
        height = height - ((height - h_min) % h_inc)

        # Set the valid ROI
        self.node_map_remote.FindNode("OffsetX").SetValue(x)
        self.node_map_remote.FindNode("OffsetY").SetValue(y)
        self.node_map_remote.FindNode("Width").SetValue(width)
        self.node_map_remote.FindNode("Height").SetValue(height)
        return x, y, width, height

    @check_camera_status
    def set_device_userid(self, display_name: str) -> str | None:
        """
        Set the display name of the camera.

        Args:
            display_name (str): the desired display name.

        Returns:
            str | None: the set display name.
        """
        max_length = self.node_map_remote.FindNode("DeviceUserID").MaximumLength()
        if len(display_name) > max_length:
            display_name = display_name[:max_length]
        self.node_map_remote.FindNode("DeviceUserID").SetValue(display_name)
        return self.node_map_remote.FindNode("DeviceUserID").Value()
