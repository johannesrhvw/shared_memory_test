import logging

from configuration.config import Config
from ids_peak import ids_peak

from camera.camera import Camera


class NoDeviceError(Exception):
    pass


class CameraManager:
    """
    A class that manages cameras using IDS peak library.
    """

    def __init__(self) -> None:
        """
        Initializes the CameraManager class.

        This method initializes the CameraManager class
        by setting up the logger,
        initializing the IDS peak library, creating a device manager instance,
        and initializing other class attributes.
        """
        self.logger = logging.getLogger(__name__)
        self.number_of_cameras = 0
        ids_peak.Library.Initialize()
        peak_version = ids_peak.Library.Version().ToString()
        self.logger.info(f"Initialized IDS peak {peak_version}")
        self.device_manager: ids_peak.DeviceManager = ids_peak.DeviceManager.Instance()
        self.cameras: dict[str, Camera] = {}
        self.serial_numbers: list[str] = []
        self.logger.info(f"CameraManager opened {self.number_of_cameras} devices.")

    def open_device(self, serial_number: str) -> Camera | None:
        # update the devices list
        self.device_manager.Update()
        # check if any devices are available
        if self.device_manager.Devices().empty():
            self.logger.error("No camera devices found.")
            raise NoDeviceError("Could not find any ids_peak devices.")
        # open all devices
        device_descriptor: ids_peak.DeviceDescriptor
        for device_descriptor in self.device_manager.Devices():
            if device_descriptor.IsOpenable() and device_descriptor.SerialNumber() == serial_number:
                # create a camera object from the device descriptor
                return Camera(device_descriptor)
        self.logger.info(f"Could not open Camera {serial_number}.")
        return None

    def open_devices(self) -> None:
        """
        Open all available camera devices.

        Raises:
            NoDeviceError: Could not find any ids_peak devices.

        Returns:
            int: Number of opened camera devices.
        """
        # update the devices list
        self.device_manager.Update()
        # check if any devices are available
        if self.device_manager.Devices().empty():
            self.logger.error("No camera devices found.")
            raise NoDeviceError("Could not find any ids_peak devices.")
        # open all devices
        device_descriptor: ids_peak.DeviceDescriptor
        for i, device_descriptor in enumerate(self.device_manager.Devices()):
            if device_descriptor.IsOpenable():
                # create a camera object from the device descriptor
                camera = Camera(device_descriptor)
                # set userid for reference
                self.cameras[camera.serial_number] = camera
                self.serial_numbers.append(camera.serial_number)
                self.logger.info(f"Opened device {camera.serial_number}.")
            self.number_of_cameras = i + 1
        self.serial_numbers.sort()

    def close(self) -> None:
        """
        Close the ids_peak library and all corresponding objects.
        """
        ids_peak.Library.Close()

    def setup_devices(self, config: Config) -> None:
        """
        Setup the camera devices with the given configuration.

        Args:
            cam_config (CameraConfig): Configuration object
            for the camera devices.

        Raises:
            ValueError: When keystring is missing in cam_config.
        """
        serial_numbers = config.serial_numbers
        if config.serial_numbers.sort() != self.serial_numbers.sort():
            self.logger.error("Serial numbers in config file " "do not match with opened cameras.")
            return
        for number in serial_numbers:
            try:
                camera = self.cameras.get(number)
            except (KeyError, ValueError):
                self.logger.exception(f"There is no camera {number} in " f"opened cameras {self.serial_numbers}. ")
                continue
            try:
                settings = config.camera_configs.get(number)
            except KeyError:
                self.logger.exception(f"Missing settings for camera {number}.")
                continue
            if camera is not None and settings is not None:
                camera.apply_settings(settings)
        self.maximum_throughput = self.calculate_maximum_throughput()
        self.set_throughput(self.maximum_throughput)

    def prepare_data_stream(self, serial_number: str) -> ids_peak.DataStream | None:
        try:
            cam = self.cameras.get(serial_number)
        except KeyError:
            self.logger.exception(
                f"Cannot get camera " f"{serial_number} in manager " f"with cameras {self.serial_numbers}."
            )
            return None
        if cam is None or cam.serial_number != serial_number:
            self.logger.error(f"Cannot get camera {serial_number} in manager " f"with cameras {self.serial_numbers}.")
            return None
        return cam.prepare_acquisition_start()

    def prepare_data_streams(self) -> dict[str, ids_peak.DataStream]:
        serial_datastreams = {}
        for number in self.serial_numbers:
            data_stream = self.prepare_data_stream(number)
            if data_stream is not None:
                serial_datastreams[number] = data_stream
        return serial_datastreams

    def calculate_maximum_throughput(self) -> int:
        """
        Calculate the maximum throughput for all connected cameras.

        Raises:
            NoDeviceError: When 0 cameras are opened.

        Returns:
            int: The maximum throughput per camera.
        """
        value: int = 0
        if self.number_of_cameras > 0:
            for number in self.serial_numbers:
                try:
                    ret = self.cameras[number].get_maximum_throughput()
                except KeyError:
                    self.logger.exception(
                        f"Cannot get camera {number} in " f"manager with " f"cameras {self.serial_numbers}."
                    )
                    return 0
                if ret is not None:
                    value += ret
            return int(((value / self.number_of_cameras) / self.number_of_cameras) * 0.6) if value is not None else 0
        self.logger.error("Cannot get maximum throughput for 0 devices.")
        raise NoDeviceError("Cannot get maximum throughput for 0 devices.")

    def set_throughput(self, throughput: int) -> None:
        """
        Set the throughput for all connected cameras.

        Args:
            throughput (int): Desired throughput value.
        """
        for cam in self.cameras.values():
            value = cam.set_throughput(throughput)
            self.logger.info(f"Set throughput to {value} for camera " f"(desired value {throughput}).")
