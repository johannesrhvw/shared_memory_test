import logging

from configuration.config import CameraConfig
from ids_peak_ipl import ids_peak_ipl


class PixelFormatError(Exception):
    pass


class FrameCorrectionError(Exception):
    pass


class FrameCorrection:
    def __init__(self, config: CameraConfig) -> None:
        """
        Initializes the FrameProcessor object.
        """
        process_settings = config.post_processing_settings
        self.ipl_gain = ids_peak_ipl.Gain()
        self.ipl_gamma_correction = ids_peak_ipl.GammaCorrector()
        self.ipl_sharpness = ids_peak_ipl.Sharpness()
        self.ipl_color_corrector = ids_peak_ipl.ColorCorrector()
        self.ipl_color_correction_factors = ids_peak_ipl.ColorCorrectionFactors()
        self.set_gain(
            process_settings.gain_red,
            process_settings.gain_green,
            process_settings.gain_blue,
        )
        self.set_gamma_correction(process_settings.gamma)
        self.set_color_correction(process_settings.color_matrix, process_settings.saturation)
        self.logger = logging.getLogger(__name__)

    def set_gain(self, r_gain: float, g_gain: float, b_gain: float) -> ids_peak_ipl.Gain:
        """
        Set the gain values for the red, green, and blue channels.

        Args:
            r_gain (float): red channel gain value.
            g_gain (float): green channel gain value.
            b_gain (float): blue channel gain value.

        Returns:
            ids_peak_ipl.Gain: Set ids_peak_ipl Gain Object.
        """
        # Retrieve maximum gain values
        r_max = self.ipl_gain.RedGainMax()
        g_max = self.ipl_gain.GreenGainMax()
        b_max = self.ipl_gain.BlueGainMax()

        # Retrieve minimum gain values
        r_min = self.ipl_gain.RedGainMin()
        g_min = self.ipl_gain.GreenGainMin()
        b_min = self.ipl_gain.BlueGainMin()

        # Clamp the gain values between their respective min and max
        r_gain = clamp(r_gain, r_min, r_max)
        g_gain = clamp(g_gain, g_min, g_max)
        b_gain = clamp(b_gain, b_min, b_max)

        # Set the new clamped gain values
        self.ipl_gain.SetRedGainValue(r_gain)
        self.ipl_gain.SetGreenGainValue(g_gain)
        self.ipl_gain.SetBlueGainValue(b_gain)
        return self.ipl_gain

    def apply_gain_inplace(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image:
        """
        Apply the gain values to the frame in place.
        Args:
            frame (ids_peak_ipl.Image): The frame to apply the gain to.

        Raises:
            PixelFormatError: When the pixel format is not supported.

        Returns:
            ids_peak_ipl.Image: Processed frame.
        """
        if self.ipl_gain.IsPixelFormatSupported(frame.PixelFormat().PixelFormatName()):
            self.ipl_gain.ProcessInPlace(frame)
            return frame
        raise PixelFormatError(f"Pixel format " f"{frame.PixelFormat().PixelFormatName()} " f"not supported.")

    def apply_gain(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image:
        """
        Apply the gain values to the frame.

        Args:
            frame (ids_peak_ipl.Image): The frame to apply the gain to.

        Raises:
            PixelFormatError: When the pixel format is not supported.

        Returns:
            ids_peak_ipl.Image: Processed frame.
        """
        if self.ipl_gain.IsPixelFormatSupported(frame.PixelFormat().PixelFormatName()):
            return self.ipl_gain.Process(frame)

        raise PixelFormatError(f"Pixel format " f"{frame.PixelFormat().PixelFormatName()} " f"not supported.")

    def set_gamma_correction(self, gamma: float) -> ids_peak_ipl.GammaCorrector:
        """
        Set value of the gamma correction.

        Args:
            gamma (float): Correction value.

        Returns:
            ids_peak_ipl.GammaCorrector: ids_peak_ipl GammaCorrector object.
        """
        min_gamma = self.ipl_gamma_correction.GammaCorrectionMin()
        max_gamma = self.ipl_gamma_correction.GammaCorrectionMax()
        gamma = clamp(gamma, min_gamma, max_gamma)
        self.ipl_gamma_correction.SetGammaCorrectionValue(gamma)
        return self.ipl_gamma_correction

    def apply_gamma_correction_inplace(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image:
        """
        Apply the gamma correction in place.

        Args:
            frame (ids_peak_ipl.Image): Frame to apply the gamma correction to.

        Raises:
            PixelFormatError: When the pixel format is not supported.

        Returns:
            ids_peak_ipl.Image: Processed frame.
        """
        if self.ipl_gamma_correction.IsPixelFormatSupported(frame.PixelFormat().PixelFormatName()):
            self.ipl_gamma_correction.ProcessInPlace(frame)
            return frame
        raise PixelFormatError(
            f"Pixel format " f"{(frame.PixelFormat().PixelFormatName().ToString())} " f"not supported."
        )

    def apply_gamma_correction(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image:
        """
        Apply the gamma correction.

        Args:
            frame (ids_peak_ipl.Image): Frame to apply the gamma correction to.

        Raises:
            PixelFormatError: When the pixel format is not supported.

        Returns:
            ids_peak_ipl.Image: Processed frame.
        """
        if self.ipl_gamma_correction.IsPixelFormatSupported(frame.PixelFormat().PixelFormatName()):
            return self.ipl_gamma_correction.Process(frame)
        raise PixelFormatError(f"Pixel format " f"{frame.PixelFormat().PixelFormatName()}" f" not supported.")

    def set_color_correction(
        self, correction_dict: tuple[float, ...], saturation: float
    ) -> ids_peak_ipl.ColorCorrectionFactors:
        """
        Set the color correction factors.

        Args:
            correction_dict (dict[str, float]): Dictionary containing factorRR,
            factorRG, factorRB, factorGR, factorGG,
            factorGB, factorBR, factorBG, factorBB, and saturation keys
            with corresponding float values.

        Returns:
            ids_peak_ipl.ColorCorrectionFactors:
            ids_peak_ipl ColorCorrectionFactors object.
        """
        self.ipl_color_correction_factors.factorRR = correction_dict[0]
        self.ipl_color_correction_factors.factorRG = correction_dict[1]
        self.ipl_color_correction_factors.factorRB = correction_dict[2]
        self.ipl_color_correction_factors.factorGR = correction_dict[3]
        self.ipl_color_correction_factors.factorGG = correction_dict[4]
        self.ipl_color_correction_factors.factorGB = correction_dict[5]
        self.ipl_color_correction_factors.factorBR = correction_dict[6]
        self.ipl_color_correction_factors.factorBG = correction_dict[7]
        self.ipl_color_correction_factors.factorBB = correction_dict[8]
        self.ipl_color_corrector.SetColorCorrectionFactors(self.ipl_color_correction_factors)
        min_saturation = self.ipl_color_corrector.SaturationMin()
        max_saturation = self.ipl_color_corrector.SaturationMax()
        saturation = clamp(saturation, min_saturation, max_saturation)
        self.ipl_color_corrector.SetSaturation(saturation)
        return self.ipl_color_correction_factors

    def apply_color_correction_inplace(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image:
        """
        Apply the color correction in place.

        Args:
            frame (ids_peak_ipl.Image): Frame to apply the color correction to.

        Raises:
            PixelFormatError: When pixel format is not supported.

        Returns:
            ids_peak_ipl.Image: Processed frame.
        """
        if self.ipl_color_corrector.IsPixelFormatSupported(frame.PixelFormat().PixelFormatName()):
            self.ipl_color_corrector.ProcessInPlace(frame)
            return frame

        raise PixelFormatError(f"Pixel format " f"{frame.PixelFormat().PixelFormatName()} " f"not supported.")

    def apply_color_correction(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image:
        """
        Apply the color correction.

        Args:
            frame (ids_peak_ipl.Image): Frame to apply the color correction to.

        Raises:
            PixelFormatError: When pixel format is not supported.

        Returns:
            ids_peak_ipl.Image: Processed frame.
        """
        if self.ipl_color_corrector.IsPixelFormatSupported(frame.PixelFormat().PixelFormatName()):
            return self.ipl_color_corrector.Process(frame)
        raise PixelFormatError(f"Pixel format " f"{frame.PixelFormat().PixelFormatName()} " f"not supported.")

    def process_image_inplace(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image | None:
        """
        Applies all the processing steps to the frame in place.

        Args:
            frame (ids_peak_ipl.Image): Frame to process.

        Returns:
            ids_peak_ipl.Image | None: Processed frame.
        """
        converted_frame = frame
        try:
            converted_frame = self.apply_gain_inplace(converted_frame)

            converted_frame = self.apply_gamma_correction_inplace(converted_frame)

            converted_frame = converted_frame.ConvertTo(ids_peak_ipl.PixelFormatName_BGR8)

            converted_frame = self.apply_color_correction_inplace(converted_frame)

        except PixelFormatError:
            self.logger.exception("Failed to process image.")
            return None
        return converted_frame

    def process_image(self, frame: ids_peak_ipl.Image) -> ids_peak_ipl.Image | None:
        """
        Applies all the processing steps to the frame.

        Args:
            frame (ids_peak_ipl.Image): Frame to process.

        Returns:
            ids_peak_ipl.Image | None: Processed frame.
        """
        try:
            gain = self.apply_gain(frame)
            gamma = self.apply_gamma_correction(gain)
            converted_frame = gamma.ConvertTo(ids_peak_ipl.PixelFormatName_BGR8)
        except PixelFormatError:
            self.logger.exception("Failed to process image.")
            return None
        return self.apply_color_correction(converted_frame)


def clamp(in_value: float, min_value: float, max_value: float) -> float:
    """
    Clamps the input value between the minimum and maximum values.

    Args:
        in_value (float): Value to clamp.
        min_value (float): Lower border.
        max_value (float): Upper border.

    Returns:
        float: Clamped value.
    """
    return max(min_value, min(in_value, max_value))
