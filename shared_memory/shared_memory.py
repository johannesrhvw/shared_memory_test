import logging
import multiprocessing
import sys
import time
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from multiprocessing.synchronize import Semaphore

import cv2
import numpy as np
from camera.acquisition_worker import AcquisitionWorker
from camera.camera_manager import CameraManager
from camera.file_frame_provider import FileFrameProvider
from camera.frame_correction import FrameCorrection
from configuration.config import Config
from configuration.logging_setup import setup_logger


CONFIG_FILE = "config/config.json"


def file_process(config: Config, file: str, shared_memory: list[SharedMemory],
                 semaphores: list[Semaphore], init_event: Event, init_memory_index: int = 0):
    logger = logging.getLogger(__name__)
    setup_logger("logs/file_process.log")
    # lock memory 0 for setup
    semaphores[init_memory_index].acquire()
    # init camera
    cam_config = config.camera_configs[config.serial_numbers[1]]
    worker = FileFrameProvider(file, cam_config)
    # init memory 0 for setup
    worker_memory_array: np.ndarray = np.ndarray(
        (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
        dtype=np.uint8,
        buffer=shared_memory[init_memory_index].buf,
    )
    logger.debug(f"Setup shared memory, {init_memory_index}.")
    current_memory_index = init_memory_index
    # get init image
    results = worker.run()
    # set memory to init image
    worker_memory_array[:] = results[0]
    init_event.set()
    logger.debug("Set the init event to signal finished setup of file_process, waiting for 0.5 seconds.")
    time.sleep(0.5)
    memory_access_count = 0
    while True:
        start = time.time()
        if current_memory_index == 0:
            memory_access_count += 1
            if semaphores[1].acquire(timeout=0.001):
                current_memory_index = 1
                logger.debug("Acquired semaphore 1.")
                semaphores[0].release()
                logger.debug("Released semaphore 0, switch complete.")
                worker_memory_array = np.ndarray(
                    (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
                    dtype=np.uint8,
                    buffer=shared_memory[1].buf,
                )
                memory_access_count = 0
        elif current_memory_index == 1:
            memory_access_count += 1
            if semaphores[0].acquire(timeout=0.001):
                logger.debug("Acquired semaphore 0.")
                current_memory_index = 0
                semaphores[1].release()
                logger.debug("Released semaphore 1, switch complete.")
                worker_memory_array = np.ndarray(
                    (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
                    dtype=np.uint8,
                    buffer=shared_memory[1].buf,
                )
                memory_access_count = 0
        results = worker.run()
        frame_number = results[1]
        worker_memory_array[:] = results[0]
        logger.debug(f"Filled memory {current_memory_index} with frame {frame_number}.")
        # time.sleep(0.01)
        end = time.time()
        logger.debug(
            f"Finished filling and switching memory {current_memory_index}, took {end - start} seconds, "
            f"accessed {memory_access_count} times."
        )


def camera_process(
    config: Config,
    shared_memory: list[SharedMemory],
    semaphores: list[Semaphore],
    init_event: Event,
    init_memory_index: int = 0,
):
    logger = logging.getLogger(__name__)
    setup_logger("logs/camera_process.log")
    camera_manager = CameraManager()
    cam_config = config.camera_configs[config.serial_numbers[1]]
    camera = camera_manager.open_device(config.serial_numbers[1])
    camera.apply_settings(cam_config)
    frame_processor = FrameCorrection(cam_config)
    data_stream = camera.prepare_acquisition_start()
    worker = AcquisitionWorker(data_stream, frame_processor)
    logger.debug("Opened device and setup worker.")
    worker_memory_array: np.ndarray = np.ndarray(
        (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
        dtype=np.uint8,
        buffer=shared_memory[init_memory_index].buf,
    )
    logger.debug(f"Setup shared memory, {init_memory_index}.")
    current_memory_index = init_memory_index
    results = worker.run()
    worker_memory_array[:] = results[0]
    semaphores[current_memory_index].release()
    logger.debug(f"Released semaphore {current_memory_index} for init.")
    current_memory_index = 1 if init_memory_index == 0 else 0
    semaphores[current_memory_index].acquire()
    logger.debug(f"Acquired semaphore {current_memory_index} for init, waiting 0.5 seconds.")
    init_event.set()
    time.sleep(0.5)
    memory_access_count = 0
    while True:
        start = time.time()
        results = worker.run()
        frame_number = results[1]
        worker_memory_array[:] = results[0]
        logger.debug(f"Filled memory {current_memory_index} with frame {frame_number}.")
        if current_memory_index == 0:
            memory_access_count += 1
            if semaphores[1].acquire(timeout=0.001):
                current_memory_index = 1
                logger.debug("Acquired semaphore 1.")
                semaphores[0].release()
                logger.debug("Released semaphore 0, switch complete.")
                worker_memory_array = np.ndarray(
                    (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
                    dtype=np.uint8,
                    buffer=shared_memory[1].buf,
                )
                memory_access_count = 0
        elif current_memory_index == 1:
            memory_access_count += 1
            if semaphores[0].acquire(timeout=0.001):
                logger.debug("Acquired semaphore 0.")
                current_memory_index = 0
                semaphores[1].release()
                logger.debug("Released semaphore 1, switch complete.")
                worker_memory_array = np.ndarray(
                    (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
                    dtype=np.uint8,
                    buffer=shared_memory[1].buf,
                )
                memory_access_count = 0
        end = time.time()
        logger.debug(
            f"Finished filling and switching memory {current_memory_index}, took {end - start} seconds, "
            f"accessed {memory_access_count} times."
        )


def consumer_process(
    config: Config,
    shared_memory: list[SharedMemory],
    semaphores: list[Semaphore],
    out_queue: Queue,
    init_event: Event,
    init_memory_index: int = 0,
):
    cam_config = config.camera_configs[config.serial_numbers[1]]
    logger = logging.getLogger(__name__)
    setup_logger("logs/consumer_process.log")
    # get the semaphore for the initial memory
    semaphores[init_memory_index].acquire(block=True)
    logger.debug(f"Acquired semaphore {init_memory_index} for init.")
    # set memory 0 for init
    consumer_memory_array: np.ndarray = np.ndarray(
        (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
        dtype=np.uint8,
        buffer=shared_memory[init_memory_index].buf,
    )
    current_memory_index = init_memory_index
    # copy data of shared memory
    image = consumer_memory_array.copy()
    # put first image index = 0 still
    out_queue.put(image)
    # signal finished setup
    init_event.set()
    while True:
        start = time.time()
        if current_memory_index == 0:
            semaphores[0].release()
            if semaphores[1].acquire(block=True):
                logger.debug("Acquired semaphore 1.")
                logger.debug("Released semaphore 0.")
                current_memory_index = 1
                consumer_memory_array = np.ndarray(
                    (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
                    dtype=np.uint8,
                    buffer=shared_memory[1].buf,
                )
            else:
                semaphores[0].acquire()
        elif current_memory_index == 1:
            semaphores[1].release()
            if semaphores[0].acquire(block=True):
                logger.debug("Acquired semaphore 0.")
                logger.debug("Released semaphore 1.")
                current_memory_index = 0
                consumer_memory_array = np.ndarray(
                    (cam_config.device_settings.roi_height, cam_config.device_settings.roi_width, 3),
                    dtype=np.uint8,
                    buffer=shared_memory[1].buf,
                )
            else:
                semaphores[1].acquire()
        image = consumer_memory_array.copy()
        logger.debug(f"Got image from shared memory {current_memory_index}.")
        # time.sleep(0.)
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        out_queue.put(image)
        time.sleep(0.1)
        end = time.time()
        logger.debug(f"Finished processing memory {current_memory_index}, took {end - start} seconds.")


def display_process(queue: Queue):
    while True:
        item = queue.get()
        if isinstance(item, np.ndarray):
            cv2.imshow("Display", item)
            cv2.waitKey(1)


def main_process_single():
    multiprocessing.set_start_method("spawn")
    context = multiprocessing.get_context()
    config = Config(CONFIG_FILE)
    byte_size_memory = (
        config.camera_configs[config.serial_numbers[0]].device_settings.roi_height
        * config.camera_configs[config.serial_numbers[0]].device_settings.roi_width
        * 3
    )
    init_event = context.Event()
    queue = Queue()
    try:
        memory_test = SharedMemory(name="image_memory_0", create=False)
        memory_test.close()
        memory_test.unlink()
    except:
        pass
    try:
        memory_test = SharedMemory(name="image_memory_1", create=False)
        memory_test.close()
        memory_test.unlink()
    except:
        pass
    memory_0 = SharedMemory(name="image_memory_0", create=True, size=byte_size_memory)
    semaphore_0 = context.Semaphore(value=1)
    memory_1 = SharedMemory(name="image_memory_1", create=True, size=byte_size_memory)
    semaphore_1 = context.Semaphore(value=1)
    memory = [memory_0, memory_1]
    semaphores = [semaphore_0, semaphore_1]
    file_process_ = context.Process(
        name="camera_process", target=file_process, args=(config, "data/cam2.avi", memory, semaphores, init_event)
    )
    consumer_process_ = context.Process(
        name="consumer_process", target=consumer_process, args=(config, memory,
                                                            semaphores, queue, init_event)
    )
    display_process_ = context.Process(name="display_process", target=display_process, args=(queue,))
    file_process_.daemon = True
    file_process_.daemon = True
    display_process_.daemon = True
    file_process_.start()
    init_event.wait()
    init_event.clear()
    consumer_process_.start()
    init_event.wait()
    init_event.clear()
    display_process_.start()
    time.sleep(100)
    file_process_.terminate()
    consumer_process_.terminate()
    display_process_.terminate()
    file_process_.join()
    consumer_process_.join()
    display_process_.join()
    memory_0.close()
    memory_1.close()
    memory_0.unlink()
    memory_1.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main_process_single())
