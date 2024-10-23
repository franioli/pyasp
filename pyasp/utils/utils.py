import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def add_directory_to_path(directory: str | Path):
    """
    Add a directory to the PATH environment variable.

    Args:
        directory (str): The directory to add to the PATH.
    """
    if not isinstance(directory, (str, Path)):
        raise TypeError("directory must be a string or Path object")
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    os.environ["PATH"] = f"{directory}:{os.environ['PATH']}"


def add_asp_binary(path: Path) -> bool:
    """
    Add the Ames Stereo Pipeline binaries to the PATH.

    Args:
        path (Path): The path to the ASP binaries.

    Returns:
        bool: True if the ASP binaries were added to the PATH, False otherwise.
    """
    add_directory_to_path(path)
    check_asp_binary()


def check_asp_binary():
    """
    Check if the Ames Stereo Pipeline binaries are in the PATH.

    Returns:
        bool: True if the ASP binaries are in the PATH, False otherwise.
    """
    if shutil.which("parallel_stereo") is not None:
        return True
    else:
        return False
