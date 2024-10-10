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
    command: List[str] | str, silent: bool = False, verbose: bool = False, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a shell command, capture output in real time, and handle errors.

    Args:
        command (List[str] | str): Command and arguments to execute.
        silent (bool): Suppress output.
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

    if silent:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        subprocess.run(command, check=True, **kwargs)
        return

    if verbose:
        logger.info(f"Executing command: {command}")

    with OutputCapture(verbose=verbose):
        if verbose:
            start_time = time.perf_counter()

        try:
            kwargs["stderr"] = subprocess.PIPE
            result = subprocess.run(command, check=True, text=True, **kwargs)
            if verbose:
                logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(e.stderr)
            return

        if verbose:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.info(f"Function {command[0]} took {total_time:.4f} s.")

    return result


class Command:
    def __init__(
        self,
        cmd: str | List[str],
        name: str = "Command",
        silent: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        if isinstance(cmd, str):
            cmd = cmd_string_to_list(cmd)
        elif isinstance(cmd, list):
            pass
        else:
            raise TypeError("cmd must be a list of strings or a single string")
        self.cmd = cmd
        self.name = name
        self.silent = silent
        self.verbose = verbose
        self.kwargs = kwargs

    def __str__(self):
        return f"{self.name} {cmd_list_to_string(self.cmd)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}: {cmd_list_to_string(self.cmd)})"

    def __call__(self):
        run_command(self.cmd, silent=self.silent, verbose=self.verbose, **self.kwargs)

    def extend(self, *args, **kwargs):
        if not args and not kwargs:
            return

        # Extend the command with each element of the args (handling lists of arguments)
        for arg in args:
            if isinstance(arg, list):
                self.cmd.extend(
                    map(str, arg)
                )  # Convert list elements to strings and extend the command
            else:
                self.cmd.append(
                    str(arg)
                )  # Convert single elements to string and append to the command

        # Extend the command with additional keyword arguments (as key-value pairs)
        for key, value in kwargs.items():
            # Replace underscores with hyphens and prepend '--' for shell-style arguments
            option = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:  # Add the option if the value is True
                    self.cmd.append(option)
            elif isinstance(value, list):
                # For list values, append each item with the option
                for item in value:
                    self.cmd.extend([option, str(item)])
            else:
                # Append the option and its value for all other types
                self.cmd.extend([option, str(value)])

    def run(self):
        run_command(self.cmd, silent=self.silent, verbose=self.verbose, **self.kwargs)


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


def check_asp_binary():
    """
    Check if the Ames Stereo Pipeline binaries are in the PATH.

    Returns:
        bool: True if the ASP binaries are in the PATH, False otherwise.
    """
    try:
        subprocess.run(
            ["parallel_stereo", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    add_directory_to_path(
        Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
    )
    cmd = "ls -l ."

    print(cmd_string_to_list(cmd))

    # out = run_command(cmd, verbose=True)

    cmd = Command("parallel_stereo --version")
    cmd()
    cmd.run()

    positional_args = ["image.tif", "metadata.xml"]
    keyword_args = {"t": "rpc", "e": "rpc"}

    cmd.extend(positional_args, **keyword_args)
