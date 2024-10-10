import os
from pathlib import Path
from typing import List

from pyasp import Command, add_directory_to_path, check_asp_binary

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

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
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

        self._verbose = verbose

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - Ames Stereo Pipeline Python Wrapper"

    def parallel_stereo(
        self,
        images: List[Path | str],
        cameras: List[Path | str] = None,
        output_file_prefix: str = "cor",
        dem: str | Path = None,
        **kwargs,
    ):
        """
        Run the ASP parallel_stereo command with provided parameters.
        Refer to https://stereopipeline.readthedocs.io/en/latest/tools/parallel_stereo.html
        for more details on available options.

        Positional arguments:
            images (list of Path or str): Input image files. If using mapprojected images, a DEM must be provided.
            cameras (list of Path or str, optional): Camera files. Can be None for certain image types like ISIS (see documentation for details).
                If cameras are required for your images and not provided, a warning will be issued.
            output_file_prefix (str): The prefix for output files. It can either be a single string or a path with a subdirectory and prefix (e.g., 'output/corr').
            dem (str or Path, optional): DEM file required if using mapprojected images. Omit for non-mapprojected images.

        Keyword arguments:
            Any other parallel_stereo command-line options can be passed as keyword arguments.

        Example:
            asp = ASP("/path/to/asp/bin")
            asp.parallel_stereo(
                images=["left_map_proj.tif", "right_map_proj.tif"],
                cameras=None,
                dem="dem_file.tif",
                output_file_prefix="output/corr",
                job_size_w=2048,
                job_size_h=2048
            )
        """

        # Ensure all image files exist
        for image in images:
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")

        # Check if cameras are required and warn if not provided for non-ISIS images
        if cameras is None:
            if any("ISIS" not in str(image) for image in images):
                raise ValueError(
                    "Camera files are required for non-ISIS images. Refer to the Ames Stereo Pipeline documentation for details."
                )
        else:
            # Ensure all camera files exist
            for camera in cameras:
                if not Path(camera).exists():
                    raise FileNotFoundError(f"Camera file not found: {camera}")

            # Ensure the number of camera files matches the number of images
            if len(cameras) != len(images):
                raise ValueError(
                    "The number of camera files must match the number of images."
                )

        # Check if DEM is required for mapprojected images
        if dem is not None:
            if not Path(dem).exists():
                raise FileNotFoundError(f"DEM file not found: {dem}")
        elif any("map_proj" in str(image) for image in images):
            raise ValueError("DEM file is required for mapprojected images.")

        # Initialize the Command object with the base command
        command = Command(
            cmd="parallel_stereo", name="parallel_stereo", verbose=self._verbose
        )

        # Add optional keyword arguments
        for key, value in kwargs.items():
            # Replace short keys with long keys for session-type and entry-point
            if key == "t":
                key = "session-type"
            if key == "e":
                key = "entry-point"

            # Add options using Command's extend method
            command.extend(**{key: value})

        # Add image files
        command.extend(images)

        # Add camera files if provided
        if cameras:
            command.extend(cameras)

        # Add the output file prefix
        command.extend(output_file_prefix)

        # Add optional DEM file
        if dem:
            command.extend(dem)

        # Now that the command is constructed, run it
        command.run()

    def run_command(self, command: str | List[str]):
        """
        Run an ASP command.

        Args:
            command (str | List[str]): Command and arguments to execute.

        """
        Command(command).run()


if __name__ == "__main__":
    asp = ASP(Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin")
    # asp.run_command("parallel_stereo --help")

    images = ["demo/hrs-1.TIF", "demo/hrs-2.TIF"]
    metadata = ["demo/hrs-1.DIM", "demo/hrs-2.DIM"]
    seed_dem = None  # "demo/seed_dem.tif"

    asp.parallel_stereo(
        images=images,
        cameras=metadata,
        output_file_prefix="output/corr",
        dem=seed_dem,
        t="spot5",
        stereo_algorithm="asp_bm",
    )
