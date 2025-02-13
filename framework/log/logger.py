import logging
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ColorHandler(logging.StreamHandler):
    GRAY7 = "38;5;7"
    RED = "31"
    WHITE = "0"

    def emit(self, record):
        level_color_map = {
            logging.INFO: self.GRAY7,
            logging.ERROR: self.RED,
        }

        csi = f"{chr(27)}["  # control sequence introducer
        color = level_color_map.get(record.levelno, self.WHITE)

        print(f"{csi}{color}m{record.message}{csi}m")


def get_info_logger(format: str = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", name: str = "",
                    info_filename: str = "info.log") -> logging.Logger:
    info_logger = logging.getLogger(f"{name}_i")
    logger_formatter = logging.Formatter(format)

    info_filehandler = logging.FileHandler(os.path.join(BASE_DIR, info_filename), mode='a')
    info_filehandler.setFormatter(logger_formatter)
    info_filehandler.setLevel(logging.INFO)
    info_logger.addHandler(info_filehandler)

    info_logger.setLevel(logging.INFO)
    info_logger.addHandler(ColorHandler())

    return info_logger


def get_error_logger(format: str = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", name: str = "",
                     error_filename: str = "error.log"):
    error_logger = logging.getLogger(f"{name}_e")
    logger_formatter = logging.Formatter(format)

    error_filehandler = logging.FileHandler(os.path.join(BASE_DIR, error_filename), mode='a')
    error_filehandler.setFormatter(logger_formatter)
    error_filehandler.setLevel(logging.ERROR)
    error_logger.addHandler(error_filehandler)

    error_logger.setLevel(logging.ERROR)

    error_logger.addHandler(ColorHandler())

    return error_logger


