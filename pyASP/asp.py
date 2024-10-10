import os
from pathlib import Path
from typing import List

from pyasp import add_directory_to_path, check_asp_binary, run_command

cores = os.cpu_count()
if not cores:
    cores = 16

_threads_singleprocess = cores  # 24, 16
_threads_multiprocess = (
    _threads_singleprocess // 2 if _threads_singleprocess > 1 else 1
)  # 12, 8
_processes = _threads_multiprocess // 4 if _threads_multiprocess > 3 else 1  # 3, 2


class ASP:
    r"""
    ASAP Stereo Pipeline - Common Commands
         ___   _____  ____ 
        /   | / ___/ / __ \
       / /| | \__ \ / /_/ /
      / ___ |___/ // ____/
     /_/  |_/____//_/    
    
    Inspired from: https://github.com/AndrewAnnex/asap_stereo

    """

    defaults_ps_s0 = {
        "--processes": _processes,
        "--threads-singleprocess": _threads_singleprocess,
        "--threads-multiprocess": _threads_multiprocess,
        "--bundle-adjust-prefix": "adjust/ba",
    }

    def __init__(self, asp_bin_dir: str | Path = None):
        """
        Initialize the ASP object.

        Args:
            asp_bin_dir (str): Path to the ASP binaries directory.
        """
        if asp_bin_dir:
            add_directory_to_path(asp_bin_dir)
        if not check_asp_binary():
            raise FileNotFoundError(
                "The Ames Stereo Pipeline binaries are not in the PATH. Please add them manually with asp.add_directory_to_path()."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - Ames Stereo Pipeline Python Wrapper"

    def parallel_stereo(
        self,
        left: Path,
        right: Path,
        left_metadata: Path,
        right_metadata: Path,
        output_dir: Path,
        seed_dem: Path,
        **kwargs,
    ):
        """
        Run the ASP parallel_stereo command.

        Args:
            left (Path): Left image.
            right (Path): Right image.
            left_metadata (Path): Left image metadata.
            right_metadata (Path): Right image metadata.
            output_dir (Path): Output directory.
            seed_dem (Path): Seed DEM.
            kwargs: Additional keyword arguments to pass to subprocess.run.

        """
        command = [
            "parallel_stereo",
            str(left),
            str(right),
            str(left_metadata),
            str(right_metadata),
            str(output_dir),
            "--seed-dem",
            str(seed_dem),
        ]
        command.extend([f"{k}={v}" for k, v in kwargs.items()])
        self.run_command(command)

    def run_command(self, command: str | List[str], verbose: bool = False):
        """
        Run an ASP command.

        Args:
            command (str | List[str]): Command and arguments to execute.
            verbose (bool): Verbose output.

        """
        run_command(command, verbose=verbose)


if __name__ == "__main__":
    asp = ASP(Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin")
    asp.run_command("parallel_stereo")
