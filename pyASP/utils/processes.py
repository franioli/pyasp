import contextlib
import io
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def cmd_list_to_string(cmd_list: List[str]) -> str:
    """
    Convert a list of command arguments to a single string.

    Args:
        cmd_list (List[str]): The list of command arguments.

    Returns:
        str: The command as a single string.
    """
    if not cmd_list:
        raise ValueError("cmd_list must not be empty")
    elif not isinstance(cmd_list, list):
        raise TypeError(
            "cmd_list must be a list of strings (numbers are automatically casted to strings)"
        )
    try:
        cmd_list = [str(arg) for arg in cmd_list]
    except TypeError as e:
        logger.error(f"Failed to convert command list to string: {e}")
        raise

    return " ".join(cmd_list)


def cmd_string_to_list(cmd_str: str) -> List[str]:
    """
    Convert a command string to a list of arguments.

    Args:
        cmd_str (str): The command as a single string.

    Returns:
        List[str]: The list of command arguments.
    """
    if not isinstance(cmd_str, str):
        raise TypeError("cmd_str must be a string")
    return cmd_str.split()


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                logger.error("Failed with output:\n%s", self.out.getvalue())
        sys.stdout.flush()


def run_command(
    command: List[str] | str, verbose: bool = False, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a shell command, capture output in real time, and handle errors.

    Args:
        command (List[str] | str): Command and arguments to execute.
        verbose (bool): Verbose output
        kwargs: Additional keyword arguments to pass to subprocess.run.

    Returns:
        CompletedProcess: The command output.
    """
    if isinstance(command, str):
        command = cmd_string_to_list(command)
    elif isinstance(command, list):
        pass
    else:
        raise TypeError("command must be a list of strings or a single string")
    if verbose:
        logger.info(f"Executing command: {command}")

    with OutputCapture(verbose=verbose):
        if verbose:
            start_time = time.perf_counter()

        output = subprocess.run(command, check=True, text=True, **kwargs)

        if verbose:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.info(f"Function {command[0]} took {total_time:.4f} s.")

    return output


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


if __name__ == "__main__":
    cmd = "ls -l ."
    print(cmd_string_to_list(cmd))

    out = run_command(cmd, verbose=True)
