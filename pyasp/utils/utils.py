import contextlib
import io
import logging
import os
import shutil
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
) -> bool:
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
            return False

        if verbose:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.info(f"Function {command[0]} took {total_time:.4f} s.")

    return True


class Command:
    """
    A class to represent a command with its parameters for execution.

    Attributes:
        cmd (list): The command and its parameters.
        name (str): A name for the command, default is "Command".
        silent (bool): If True, suppress output during execution.
        verbose (bool): If True, provide detailed output during execution.
        kwargs (dict): Additional keyword arguments for command execution.
    """

    def __init__(
        self,
        cmd: str | list[str],
        name: str = "Command",
        silent: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initializes a Command instance.

        Args:
            cmd (str | list[str]): The base command as a string or a list of strings.
            name (str, optional): The name of the command. Default is "Command".
            silent (bool, optional): Suppress output if True. Default is False.
            verbose (bool, optional): Provide detailed output if True. Default is False.
            **kwargs: Additional keyword arguments for command execution.

        Raises:
            TypeError: If `cmd` is neither a string nor a list of strings.
        """
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

    def __str__(self) -> str:
        """
        Returns a string representation of the Command.

        Returns:
            str: The string representation of the Command.
        """
        return f"{cmd_list_to_string(self.cmd)}"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the Command for debugging.

        Returns:
            str: The string representation for debugging.
        """
        return f"{self.__class__.__name__}({self.name}: {self})"

    def __call__(self) -> bool:
        """
        Executes the command by calling the run_command function with the command parameters.

        Additional arguments from kwargs are passed to the run_command function.
        """
        return self.run()

    def extend(self, *args, **kwargs):
        """
        Extends the command with additional arguments.

        Args:
            *args: Positional arguments to extend the command.
            **kwargs: Keyword arguments to add as parameters to the command.
        """
        if not args and not kwargs:
            return

        # Extend the command with the positional arguments
        for arg in args:
            # Handle the case where the argument is a list: Convert list elements to strings
            if isinstance(arg, list):
                self.cmd.extend(map(str, arg))
                continue

            # Convert single elements to string and append to the command
            self.cmd.append(str(arg))

        # Extend the command with additional keyword arguments
        for key, value in kwargs.items():
            # Handle boolean flags (no value)
            if value is True:
                self.cmd.append(f"{key}")
                continue

            # Handle the case where the value is a list
            if isinstance(value, list):
                self.cmd.extend([f"{key}", *map(str, value)])
                continue

            # Add the key and value as separate arguments
            self.cmd.extend([f"{key}", str(value)])

    def run(self):
        """
        Runs the command using the run_command function.

        Additional arguments from kwargs are passed to the run_command function.
        """
        return run_command(
            self.cmd, silent=self.silent, verbose=self.verbose, **self.kwargs
        )

    def get_parameters(self) -> list[str]:
        """
        Returns a list of command parameters and their values (if any).

        Parameters are considered those that start with '--' or '-'.

        Returns:
            list[str]: A list of parameters and their associated values.
        """
        parameters = []
        skip_next = False

        for i, arg in enumerate(self.cmd):
            if skip_next:
                skip_next = False
                continue

            # Check if the argument is a parameter (starts with -- or -)
            if arg.startswith("--") or arg.startswith("-"):
                parameters.append(arg)
                # If the next argument is not a parameter, it is likely a value for this parameter
                if (
                    i + 1 < len(self.cmd)
                    and not self.cmd[i + 1].startswith("--")
                    and not self.cmd[i + 1].startswith("-")
                ):
                    parameters.append(self.cmd[i + 1])
                    skip_next = True  # Skip the next argument since it's a value

        return parameters


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
    if shutil.which("parallel_stereo") is not None:
        return True
    else:
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
