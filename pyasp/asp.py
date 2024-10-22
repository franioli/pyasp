"""
How to implement a new ASP function:
"""

import os
import re
import subprocess
import time
from abc import ABC
from pathlib import Path
from typing import List

from pyasp import Command, add_directory_to_path, check_asp_binary, logger

cores = os.cpu_count()
if not cores:
    cores = 16

_threads_singleprocess = cores  # 24, 16
_threads_multiprocess = (
    _threads_singleprocess // 2 if _threads_singleprocess > 1 else 1
)  # 12, 8
_processes = _threads_multiprocess // 4 if _threads_multiprocess > 3 else 1  # 3, 2


def check_file_exists(path: Path | str) -> bool:
    path = Path(path)
    if path.is_symlink():
        target_path = path.readlink()
        if not target_path.exists():
            raise FileNotFoundError(f"Target of symlink not found: {target_path}")
    else:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    return True


def check_asp_command(command: Command) -> bool:
    """
    Verifies that all parameters in the command are valid by comparing them to the
    valid parameters of the ASP function (retrieved via --help).

    Args:
        command (Command): The command object containing the ASP function and its parameters.

    Returns:
        bool: Returns True if all parameters are valid, raises an error otherwise.

    Raises:
        ValueError: If an invalid parameter is found.
    """

    # Extract the base ASP command (the first element of the command)
    asp_command = command.cmd

    # Run the ASP command with the --help flag to get the valid parameters
    try:
        result = subprocess.run(
            [asp_command[0], "--help"], capture_output=True, text=True
        )
        help_output = result.stdout

        # Parse the help output to extract valid parameters
        valid_parameters = extract_parameters_from_help(help_output)

        # Get the parameters from the command object
        provided_parameters = command.get_parameters()

        # Compare each provided parameter with the list of valid parameters
        for param in provided_parameters:
            if param.startswith("--") and param not in valid_parameters:
                raise ValueError(
                    f"Invalid parameter '{param}' for the command '{command}'"
                )
        return True

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute {asp_command} --help: {e}")


def extract_parameters_from_help(help_output: str) -> List[str]:
    """
    Extracts valid parameters from the help output of an ASP command.

    Args:
        help_output (str): The output of the command run with --help.

    Returns:
        List[str]: A list of valid parameters.
    """
    # Regex to match long options (parameters starting with --)
    param_regex = re.compile(r"--[a-zA-Z0-9_-]+")

    # Find all matches in the help output
    parameters = param_regex.findall(help_output)

    return parameters


def kwargs_to_asp(dict: dict) -> dict:
    """
    Convert keyword arguments to ASP command-line format.

    Args:
        dict (dict): A dictionary of keyword arguments.

    Returns:
        dict: A dictionary of keyword arguments formatted for the ASP command line.
    """
    new_dict = {}
    for key, value in dict.items():
        # Replace underscores with hyphens
        key = key.replace("_", "-")

        if len(key) == 1:  # Short keys
            if isinstance(value, bool):
                if value:
                    new_dict[f"--{key}"] = ""
            else:
                new_dict[f"-{key}"] = value
        elif len(key) > 1:  # Long keys
            if isinstance(value, bool):
                if value:
                    new_dict[f"--{key}"] = ""
            else:
                new_dict[f"--{key}"] = value
        else:
            raise ValueError(f"Invalid key: {key}")

    return new_dict


class AspStepBase(ABC):
    _command = None
    _verbose = False
    _asp_bin_dir = None
    _skip_check = False
    _elaspsed_time = None

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initialize the AspStepBase object, setting up the environment for ASP tools.

        This abstract base class provides the foundational structure for all ASP tool wrappers.
        Subclasses must implement the `bake` method to construct the specific command they represent.

        Args:
            asp_bin_dir (str or Path, optional):
                Path to the Ames Stereo Pipeline (ASP) binaries directory.
                If provided, this directory is added to the system PATH to ensure ASP executables are accessible.
            verbose (bool, optional):
                If set to True, command execution will output detailed logs. Defaults to False.

        Raises:
            FileNotFoundError:
                If the ASP binaries are not found in the specified directory or the system PATH.
        """
        if asp_bin_dir:
            self._asp_bin_dir = Path(asp_bin_dir).resolve()
            add_directory_to_path(self._asp_bin_dir)
        if not check_asp_binary():
            raise FileNotFoundError(
                "The Ames Stereo Pipeline binaries are not in the PATH. Please add them manually with asp.add_directory_to_path()."
            )

        self._verbose = verbose

    def __repr__(self) -> str:
        """
        Return a string representation of the AspStepBase object.

        Returns:
            str: A string indicating the class name and the command to be executed.
        """
        return f"{self.__class__.__name__} - '{self._command}'"

    def __call__(self, skip_check: bool = False):
        """
        Execute the constructed ASP command.

        This method runs the command that has been built using the `bake` method.
        It raises an error if the command has not been set.

        Args:
            skip_check (bool, optional):
                If set to True, the command will not be checked for valid parameters before execution. Defaults to False.

        Raises:
            ValueError: If the command has not been set by calling `bake`.
            RuntimeError: If the command execution fails.
        """
        if skip_check:
            self._skip_check = True

        if not self._skip_check:
            check_asp_command(self._command)

        logger.info(f"Running command: {self._command}")
        start_time = time.perf_counter()
        ret = self._command.run()
        if ret:
            self._elaspsed_time = time.perf_counter() - start_time
            logger.info(
                f"Command {self._command.name} completed successfully. Total time: {self._elaspsed_time:.2f} seconds"
            )
        else:
            raise RuntimeError(
                f"ASP processing step '{self._command.name}' failed [command: '{self._command}']."
            )

    @property
    def command(self) -> Command:
        """
        Return the command object.

        Returns:
            Command: The command object.
        """
        return self._command

    @property
    def elapsed_time(self) -> float:
        """
        Return the elapsed time of the last command execution.

        Returns:
            float: The elapsed time in seconds.
        """
        if self._elaspsed_time is None:
            logger.info("The step has not been executed yet.")
        return self._elaspsed_time

    @classmethod
    def from_command(cls, command: Command | str):
        """
        Create an instance of the AspStepBase class from a Command object.

        Args:
            command (Command): The Command object to convert to an AspStepBase instance.

        Returns:
            AspStepBase: An instance of the AspStepBase class.
        """
        instance = cls()

        if isinstance(command, str):
            command = Command(command)

        instance._command = command
        return instance


class ParallelStereo(AspStepBase):
    """
    A class to run the parallel_stereo software from the Ames Stereo Pipeline (ASP). https://stereopipeline.readthedocs.io/en/latest/tools/parallel_stereo.html
    """

    def __init__(
        self,
        images: List[Path | str],
        cameras: List[Path | str] = None,
        output_file_prefix: str = "cor",
        dem: str | Path = None,
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the ParallelStereo class by constructing the command directly.

        Args:
            images (List[Path or str]): Input image files.
            cameras (List[Path or str], optional): Camera files, required for non-ISIS images.
            output_file_prefix (str, optional): The prefix for output files. Defaults to "cor".
            dem (str or Path, optional): DEM file, required if using mapprojected images.
            asp_bin_dir (str or Path, optional): Path to the ASP binaries directory.
            verbose (bool, optional): If True, enables verbose output for debugging purposes.
            **kwargs: Additional `parallel_stereo` command-line options.

        Raises:
            FileNotFoundError: If any required file (images, cameras, DEM) is missing.
            ValueError: If the number of images and cameras doesn't match or DEM is missing for mapprojected images.
        """
        # Call the parent constructor to check binary existence
        super().__init__(asp_bin_dir=asp_bin_dir, verbose=verbose)

        # Ensure all image files exist
        for image in images:
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")

        # Check if cameras are required and validate them
        if cameras is None:
            if any("ISIS" not in str(image) for image in images):
                raise ValueError("Camera files are required for non-ISIS images.")
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

        # Construct the command object
        command = Command(
            cmd="parallel_stereo", name="parallel_stereo", verbose=self._verbose
        )

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

        # Add any additional optional arguments from kwargs
        command.extend(**kwargs_to_asp(kwargs))

        # Set the command for execution
        self._command = command

        # Force skip check for parallel_stereo as not all the possible parameters are available in the --help output
        logger.info(
            "Skipping command check for parallel_stereo. Some parameters may not be validated."
        )
        self._skip_check = True


class BundleAdjust(AspStepBase):
    """
    A class to run the bundle_adjust software from the Ames Stereo Pipeline (ASP). https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html

    Args:
        asp_bin_dir (str): Path to the ASP binaries directory.
        verbose (bool): If True, enables verbose output for debugging purposes.
    """

    def __init__(
        self,
        images: List[Path | str],
        cameras: List[Path | str],
        output_prefix: str = "ba_run",
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        BundleAdjust constructor to initialize command and perform file checks.

        Args:
            images (list of Path or str): List of input images.
            cameras (list of Path or str): List of input camera files associated with the images.
            output_prefix (str, optional): The prefix for output files. Defaults to "ba_run".
            asp_bin_dir (str or Path, optional): Path to ASP binaries.
            verbose (bool, optional): Enable verbose output. Defaults to False.
            **kwargs: Additional parameters to pass to the bundle_adjust command.

        Raises:
            FileNotFoundError: If any of the input image or camera files are missing.
            ValueError: If the number of images doesn't match the number of cameras.
        """
        # First call the parent class constructor to handle binary checks
        super().__init__(asp_bin_dir=asp_bin_dir, verbose=verbose)

        # Ensure all image files exist (following symlinks)
        for path in images:
            check_file_exists(path)

        # Ensure all camera files exist (following symlinks)
        for camera in cameras:
            check_file_exists(camera)

        # Ensure the number of camera files matches the number of images
        if len(cameras) != len(images):
            raise ValueError(
                "The number of camera files must match the number of images."
            )

        # Initialize the Command object
        command = Command(
            cmd="bundle_adjust", name="bundle_adjust", verbose=self._verbose
        )

        # Add image files
        command.extend(images)

        # Add camera files
        command.extend(cameras)

        # Add the output file prefix
        command.extend(**{"-o": output_prefix})

        # Add additional optional arguments from kwargs
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command

    def read_match_file(self, match_file: Path):
        """Read an ASP match file."""
        pass


class AddSpotRPC(AspStepBase):
    """
    A class to run the AddSpotRPC function. https://stereopipeline.readthedocs.io/en/latest/tools/add_spot_rpc.html
    """

    def __init__(
        self,
        input_metadata_file: str | Path,
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the AddSpotRPC class by constructing the command.

        Args:
            input_metadata_file (str or Path): Path to the input SPOT5 metadata file.
            asp_bin_dir (str or Path, optional): Path to the ASP binaries directory.
            verbose (bool, optional): If True, enables verbose output for debugging purposes.
            **kwargs: Additional `add_spot_rpc` command-line options.

        Raises:
            FileNotFoundError: If the input metadata file is missing.
        """

        # Call the parent constructor to check binary existence
        super().__init__(asp_bin_dir=asp_bin_dir, verbose=verbose)

        # Ensure the input metadata file exists
        if not Path(input_metadata_file).exists():
            raise FileNotFoundError(
                f"Input metadata file not found: {input_metadata_file}"
            )

        # Initialize the Command object with the base command
        command = Command(
            cmd="add_spot_rpc", name="add_spot_rpc", verbose=self._verbose
        )

        # Add input metadata file
        command.extend(input_metadata_file)

        # Handle the output prefix (due to a bug in ASP, must use -o flag)
        if "output_prefix" in kwargs:
            kwargs["o"] = kwargs.pop("output_prefix")

        # Add additional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        # Set the command for execution
        self._command = command


class MapProject(AspStepBase):
    """
    A class to run the mapproject function from the Ames Stereo Pipeline (ASP). https://stereopipeline.readthedocs.io/en/latest/tools/mapproject.html
    """

    def __init__(
        self,
        dem: Path | str,
        camera_image: Path | str,
        camera_model: Path | str,
        output_image: Path | str,
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the MapProject class by constructing the command.

        Args:
            dem (Path or str): The DEM file used for map projection.
            camera_image (Path or str): The camera image to be projected.
            camera_model (Path or str): The camera model associated with the camera image.
            output_image (Path or str): The output image file.
            asp_bin_dir (str or Path, optional): Path to the ASP binaries directory.
            verbose (bool, optional): If True, enables verbose output for debugging purposes.
            **kwargs: Additional `mapproject` command-line options.

        Raises:
            FileNotFoundError: If any of the required files (DEM, camera image, camera model) are missing.
        """

        # Call the parent constructor to check binary existence
        super().__init__(asp_bin_dir=asp_bin_dir, verbose=verbose)

        # Ensure all input files exist
        if not Path(dem).exists():
            raise FileNotFoundError(f"DEM file not found: {dem}")
        if not Path(camera_image).exists():
            raise FileNotFoundError(f"Camera image file not found: {camera_image}")
        if not Path(camera_model).exists():
            raise FileNotFoundError(f"Camera model file not found: {camera_model}")

        # Initialize the Command object with the base command
        command = Command(cmd="mapproject", name="mapproject", verbose=self._verbose)

        # Add positional arguments (DEM, camera image, camera model, output image)
        command.extend([dem, camera_image, camera_model, output_image])

        # Add optional keyword arguments from kwargs
        command.extend(**kwargs_to_asp(kwargs))

        # Set the command for execution
        self._command = command


class Point2dem(AspStepBase):
    """
    A class to represent the Point2DEM function in the Ames Stereo Pipeline. https://stereopipeline.readthedocs.io/en/latest/tools/point2dem.html
    """

    def __init__(
        self,
        input_file: str | Path,
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initializes a Point2dem instance and constructs the command.

        Args:
            input_file (str | Path): The input file for the Point2DEM command.
            asp_bin_dir (str | Path, optional): The directory of the ASP binaries. Default is None.
            verbose (bool, optional): If True, provide detailed output during execution. Default is False.
            **kwargs: Additional keyword arguments to be passed as parameters.

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        super().__init__(asp_bin_dir, verbose)

        # Ensure the input file exists
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Initialize the Command object with the base command
        command = Command(cmd="point2dem", name="point2dem", verbose=self._verbose)

        # Add the input file
        command.extend(input_file)

        # Add optional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        # Set the command for execution
        self._command = command


class Point2las(AspStepBase):
    """
    A class to represent the point2las function in the Ames Stereo Pipeline. https://stereopipeline.readthedocs.io/en/latest/tools/point2las.html

    """

    def __init__(
        self,
        input_file: str | Path,
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initializes a Point2Las instance and constructs the command.

        Args:
            input_file (str | Path): The input PC file for the point2las command.
            asp_bin_dir (str | Path, optional): The directory of the ASP binaries. Default is None.
            verbose (bool, optional): If True, provide detailed output during execution. Default is False.
            **kwargs: Additional keyword arguments to be passed as parameters.

        Raises:
            FileNotFoundError: If the input PC file does not exist.
        """
        super().__init__(asp_bin_dir, verbose)

        # Ensure the input file exists
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Initialize the Command object with the base command
        command = Command(cmd="point2las", name="point2las", verbose=self._verbose)

        # Add the input file
        command.extend(input_file)

        # Add optional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        # Set the command for execution
        self._command = command


class DEMGeoid(AspStepBase):
    """
    A class to represent the dem_geoid function in the Ames Stereo Pipeline.  https://stereopipeline.readthedocs.io/en/latest/tools/dem_geoid.html
    """

    def __init__(
        self,
        input_dem: str | Path,
        asp_bin_dir: str | Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initializes a DEMGeoid instance and constructs the command.

        Args:
            input_dem (str | Path): The input DEM for the dem_geoid command.
            asp_bin_dir (str | Path, optional): The directory of the ASP binaries. Default is None.
            verbose (bool, optional): If True, provide detailed output during execution. Default is False.
            **kwargs: Additional keyword arguments to be passed as parameters.

        Raises:
            FileNotFoundError: If the input DEM does not exist.
        """
        super().__init__(asp_bin_dir, verbose)

        # Ensure the input DEM exists
        if not Path(input_dem).exists():
            raise FileNotFoundError(f"Input DEM file not found: {input_dem}")

        # Initialize the Command object with the base command
        command = Command(cmd="dem_geoid", name="dem_geoid", verbose=self._verbose)

        # Add the input DEM
        command.extend(input_dem)

        # Add optional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        # Set the command for execution
        self._command = command


if __name__ == "__main__":
    add_directory_to_path(
        Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
    )

    # Run parallel_stereo command

    images = ["demo/hrs-1.TIF", "demo/hrs-2.TIF"]
    metadata = ["demo/hrs-1.DIM", "demo/hrs-2.DIM"]
    seed_dem = None  # "demo/seed_dem.tif"

    # Add Spot RPC
    asr = AddSpotRPC(
        input_metadata_file="demo/hrs-1.DIM",
        output_prefix="demo/hrs-1.txt",  # or o="demo/hrs-1.txt"
        min_height=100,
        max_height=4000,
    )
    asr()

    # Parallel Stereo
    ps = ParallelStereo(
        images=images,
        cameras=metadata,
        output_file_prefix="output/corr",
        dem=seed_dem,
        t="spot5",
        stereo_algorithm="asp_bm",
    )
    ps()

    p2d = Point2dem(
        "/home/francesco/uzh/aletsch_spot5/ASP_proc/output/corr/corr-PC.tif",
        o="output/corr-DEM.tif",
        tr=10,
        # orthoimage="/home/francesco/uzh/aletsch_spot5/ASP_proc/output/corr/corr-PC.tif",
    )
    p2d()

    p2l = Point2las(
        "/home/francesco/uzh/aletsch_spot5/ASP_proc/output/corr/corr-PC.tif",
        o="/home/francesco/uzh/aletsch_spot5/ASP_proc/output/corr/corr-pcd.las",
    )
    p2l()
