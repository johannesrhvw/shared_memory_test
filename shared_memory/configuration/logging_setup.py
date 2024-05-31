import logging
import os


def setup_logger(filename: str) -> None:
    if os.path.isfile(filename):
        os.remove(filename)
    os.open(filename, os.O_CREAT)
    root = logging.getLogger()
    file_handler = logging.FileHandler(filename, mode="a")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(threadName)s %(filename)s %(funcName)-4s %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)
