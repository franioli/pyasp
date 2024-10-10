import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import OmegaConf

from pyasp import logger


class AmesStereoPipelineError(Exception):
    """Custom exception for Ames Stereo Pipeline errors."""

    pass


class AmesStereoPipelineBase(ABC):
    def __init__(self, config_path: Path, asp_path: Path = None):
        # Load configuration
        config_path = Path(config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self._config = OmegaConf.load(config_path)

        # Add ASP binary directory to PATH
        if asp_path:
            asp_bin_dir = asp_path.resolve()
        else:
            try:
                asp_bin_dir = Path(self._config.asp_bin_dir).resolve()
            except AttributeError:
                raise AmesStereoPipelineError(
                    "ASP binary directory not provided or configured"
                )

        if asp_bin_dir.is_dir():
            os.environ["PATH"] += os.pathsep + str(asp_bin_dir)
            logger.debug(f"Added ASP binary directory to PATH: {asp_bin_dir}")
        else:
            logger.error(f"ASP binary directory does not exist: {asp_bin_dir}")
            raise AmesStereoPipelineError(
                f"ASP binary directory does not exist: {asp_bin_dir}"
            )

        # Set up paths
        self.data_dir = Path(self._config.data_dir).resolve()
        if not self.data_dir.is_dir():
            raise AmesStereoPipelineError(
                f"Data directory does not exist: {self.data_dir}"
            )

        self.output_dir = Path(self._config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def run_command(
        self, command: list, cwd: Path = None, suppress_output: bool = False
    ):
        """
        Run a shell command, capture output in real time, and handle errors.

        Args:
            command (list): Command and arguments to execute.
            cwd (Path, optional): Working directory to execute the command.

        Returns:
            str: Collected stdout output.

        Raises:
            AmesStereoPipelineError: If the command execution fails.
        """
        logger.info(f"Executing command: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ,  # Ensure updated PATH is used
            )

            stdout_lines = []
            stderr_lines = []

            # Stream stdout
            for line in iter(process.stdout.readline, ""):
                if line:
                    if not suppress_output:
                        logger.info(line.strip())
                    stdout_lines.append(line)
            process.stdout.close()

            # Stream stderr
            for line in iter(process.stderr.readline, ""):
                if line:
                    logger.warning(line.strip())
                    stderr_lines.append(line)
            process.stderr.close()

            return_code = process.wait()
            if return_code != 0:
                logger.error(f"Command failed with return code {return_code}")
                logger.error("".join(stderr_lines))
                raise AmesStereoPipelineError(
                    f"Command failed: {' '.join(command)}\nReturn Code: {return_code}\nError Output: {''.join(stderr_lines)}"
                )

            collected_stdout = "".join(stdout_lines)
            return collected_stdout

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(command)}")
            logger.error(e.stderr)
            raise AmesStereoPipelineError(f"Command failed: {' '.join(command)}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during command execution: {' '.join(command)}"
            )
            raise AmesStereoPipelineError(f"Unexpected error: {e}") from e

    def get_command_options(self, options: dict) -> list:
        """Convert a dictionary of options to a list of command line arguments."""
        command_options = []
        for key, value in options.items():
            if value is not None:
                command_options.extend([f"--{key}", str(value)])
        return command_options

    def copy_file(self, source: Path, destination: Path):
        """Copy file from source to destination."""
        try:
            shutil.copy(source, destination)
            logger.debug(f"Copied {source} to {destination}")
        except Exception as e:
            logger.error(f"Failed to copy {source} to {destination}: {e}")
            raise AmesStereoPipelineError(
                f"Failed to copy {source} to {destination}"
            ) from e

    @abstractmethod
    def process_pair(self, pair_name: str):
        """Process a single stereo pair."""
        pass

    # @abstractmethod
    # def process_all_pairs(self):
    #     """Process all stereo pairs."""
    #     pass


if __name__ == "__main__":
    pass
