import json

import numpy as np


class Config:
    def __init__(self, file: str, max_cam_count: int = 0) -> None:
        with open(file) as f:
            data = json.load(f)
        self.number_of_cameras = 0
        self.serial_numbers: list[str] = []
        self.camera_configs: dict[str, CameraConfig] = {}
        self.general_config = GeneralConfig(data["general"])
        try:
            while max_cam_count == 0 or self.number_of_cameras < max_cam_count:
                camera_conf_data = data[f"camera_{self.number_of_cameras}"]
                camera_config = CameraConfig(camera_conf_data, self.general_config)
                camera_config.general_config = self.general_config
                self.camera_configs[camera_config.serial_number] = camera_config
                self.serial_numbers.append(camera_config.serial_number)
                self.number_of_cameras += 1
        except KeyError:
            return


class GeneralConfig:
    class ConsumerConfig:
        class YoloSettings:
            def __init__(self, data: dict) -> None:
                self.model_path = data["model_path"]
                self.input_width = data["input_width"]
                self.input_height = data["input_height"]
                self.confidence_threshold = data["confidence_threshold"]
                self.iou_threshold = data["iou_threshold"]
                self.half = data["half"]

        class YoloSegSettings(YoloSettings):
            def __init__(self, data: dict) -> None:
                super().__init__(data)
                self.retina_masks = data["retina_masks"]

        class YoloClsfySettings(YoloSettings):
            def __init__(self, data: dict) -> None:
                super().__init__(data)
                self.batch_size = data["batch_size"]
                self.number_of_workers = data["number_of_workers"]

        class NorfairTrackingSettings:
            def __init__(self, data: dict) -> None:
                self.track_points = data["track_points"]
                self.distance_function = data["distance_function"]
                self.distance_threshold = data["distance_threshold"]
                self.hit_counter_max = data["hit_counter_max"]
                self.initialization_delay = data["initialization_delay"]
                self.pointwise_hit_counter_max = data["pointwise_hit_counter_max"]
                self.detection_threshold = data["detection_threshold"]
                self.past_detections_length = data["past_detections_length"]

        class ObjectROIFilterSettings:
            def __init__(self, data: dict) -> None:
                self.margin_top = data["margin_top"]
                self.margin_bottom = data["margin_bottom"]
                self.margin_left = data["margin_left"]
                self.margin_right = data["margin_right"]

        def __init__(self, data: dict) -> None:
            self.yolo_seg = self.YoloSegSettings(data["yolo_segmentation_inference_settings"])

            self.yolo_clsfy = self.YoloClsfySettings(data["yolo_classification_inference_settings"])

            self.norfair_tracker = self.NorfairTrackingSettings(data["norfair_tracker_settings"])

            self.object_roi_filter = self.ObjectROIFilterSettings(data["object_region_of_interest"])
            self.consumer_interval = data["consumer_interval"]

    class SorterConfig:
        def __init__(self, data: dict) -> None:
            self.sort_interval = data["sort_interval"]
            self.paddle_width = data["paddle_width"]
            self.sort_line = data["sort_line"]
            self.ip_address = data["ip_address"]
            self.active_valve_time = data["active_valve_time"]
            if self.active_valve_time < self.sort_interval:
                raise ValueError(
                    f"Clock period {self.sort_interval}s exceeds valve " f"active time {self.active_valve_time}s."
                )

    def __init__(self, data: dict) -> None:
        self.consumer_config = self.ConsumerConfig(data["consumer"])
        self.sorter_config = self.SorterConfig(data["sorter"])


class CameraConfig:
    class DeviceSettings:
        def __init__(self, data: dict) -> None:
            self.exposure = data["exposure"]
            self.fps = data["fps"]
            self.roi_width = data["roi_width"]
            self.roi_height = data["roi_height"]
            self.x_offset = data["x_offset"]
            self.y_offset = data["y_offset"]
            self.ids_cset_file = data["ids_cset_file"]

    class PostProcessingSettings:
        def __init__(self, data: dict) -> None:
            self.gain_green = data["gain_green"]
            self.gain_red = data["gain_red"]
            self.gain_blue = data["gain_blue"]
            col_mat_data = data["color_matrix"]
            self.color_matrix = (
                col_mat_data["RedRed"],
                col_mat_data["RedGreen"],
                col_mat_data["RedBlue"],
                col_mat_data["GreenRed"],
                col_mat_data["GreenGreen"],
                col_mat_data["GreenBlue"],
                col_mat_data["BlueRed"],
                col_mat_data["BlueGreen"],
                col_mat_data["BlueBlue"],
            )
            self.saturation = data["saturation"]
            self.gamma = data["gamma"]

    class FOVTransformerSettings:
        def __init__(self, data: dict) -> None:
            target_data = data["target_fov"]
            self.target_x_offset = target_data["x_offset"]
            self.target_y_offset = target_data["y_offset"]
            self.target_width = target_data["width"]
            self.target_height = target_data["height"]
            self.target_fov = np.array(
                [
                    [self.target_x_offset, self.target_y_offset],
                    [
                        self.target_x_offset + self.target_width,
                        self.target_y_offset,
                    ],
                    [
                        self.target_x_offset + self.target_width,
                        self.target_y_offset + self.target_height,
                    ],
                    [
                        self.target_x_offset,
                        self.target_y_offset + self.target_height,
                    ],
                ]
            )
            source_data = data["source_fov"]
            self.source_x_offset = source_data["x_offset"]
            self.source_y_offset = source_data["y_offset"]
            self.source_width = source_data["width"]
            self.source_height = source_data["height"]
            self.source_fov = np.array(
                [
                    [self.source_x_offset, self.source_y_offset],
                    [
                        self.source_x_offset + self.source_width,
                        self.source_y_offset,
                    ],
                    [
                        self.source_x_offset + self.source_width,
                        self.source_y_offset + self.source_height,
                    ],
                    [
                        self.source_x_offset,
                        self.source_y_offset + self.source_height,
                    ],
                ]
            )

    def __init__(self, data: dict, general_config: GeneralConfig) -> None:
        self.general_config: GeneralConfig = general_config
        self.position_id = data["camera_position_id"]
        self.serial_number = data["camera_serial_number"]
        self.device_settings = self.DeviceSettings(data["camera_device_settings"])

        self.post_processing_settings = self.PostProcessingSettings(data["camera_postprocessing_settings"])

        self.fov_transformer_settings = self.FOVTransformerSettings(data["camera_field_of_view_transformer_settings"])
