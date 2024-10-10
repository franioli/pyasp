from .utils.logger import change_logger_level, get_logger, setup_logger  # noqa: F401
from .utils.timer import Timer, timeit  # noqa: F401
from .utils.utils import *  # noqa: F401, F403

__version__ = "0.0.2"

logger = setup_logger(name="pyasp", log_level="info")
timer = Timer(logger=logger)

# Try to run an AmesStereoPipeline command. If it fails, ask the user to manually add the ASP binaries to the PATH with asp.add_directory_to_path()
if not check_asp_binary():
    logger.warning(
        "The AmesStereoPipeline binaries are not in the PATH environmental variable. Please add them manually with asp.add_directory_to_path()."
    )
