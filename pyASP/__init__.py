from .utils.logger import change_logger_level, get_logger, setup_logger  # noqa: F401
from .utils.timer import Timer, timeit  # noqa: F401

__version__ = "1.0.0"

logger = setup_logger(name="pyasp", log_level="info")
timer = Timer(logger=logger)