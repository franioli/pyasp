from .utils.logger import change_logger_level, get_logger, setup_logger  # noqa: F401
from .utils.shell import Command  # noqa: F401
from .utils.timer import Timer, timeit  # noqa: F401
from .utils.utils import (  # noqa: F401
    add_asp_binary,
    add_directory_to_path,
    check_asp_binary,
)

__version__ = "0.0.2"

logger = setup_logger(name="pyasp", log_level="info", log_folder="./logs")
timer = Timer(logger=logger)

# Try to run an AmesStereoPipeline command. If it fails, ask the user to manually add the ASP binaries to the PATH with asp.add_directory_to_path()
if not check_asp_binary():
    logger.warning(
        "The AmesStereoPipeline binaries not found. Please add them to the PATH environmental variable with pyasp.add_asp_binary('path/to/asp/binaries')."
    )
