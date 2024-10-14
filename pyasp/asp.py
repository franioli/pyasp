import os
import re
import subprocess
from abc import ABC, abstractmethod
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
            [asp_command[0], "--help"], capture_output=True, text=True, check=True
        )
        help_output = result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute {asp_command} --help: {e}")

    # Parse the help output to extract valid parameters
    valid_parameters = extract_parameters_from_help(help_output)

    # Get the parameters from the command object
    provided_parameters = command.get_parameters()

    # Compare each provided parameter with the list of valid parameters
    for param in provided_parameters:
        if param.startswith("--") and param not in valid_parameters:
            raise ValueError(
                f"Invalid parameter '{param}' for the command '{asp_command}'"
            )

    return True


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


class ASPFunctionBase(ABC):
    _command = None
    _verbose = False
    _asp_bin_dir = None

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initialize the ASPFunctionBase object, setting up the environment for ASP tools.

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
        Return a string representation of the ASPFunctionBase object.

        Returns:
            str: A string indicating the class name and the command to be executed.
        """
        return f"{self.__class__.__name__} - '{self._command}'"

    def __call__(self):
        """
        Execute the constructed ASP command.

        This method runs the command that has been built using the `bake` method.
        It raises an error if the command has not been set.

        Raises:
            ValueError: If the command has not been set by calling `bake`.
            RuntimeError: If the command execution fails.
        """
        if not self._command:
            raise ValueError("Command not set. Please call bake() first.")
        check_asp_command(self._command)
        logger.info(f"Running command: {self._command}")
        self._command.run()
        logger.info(f"Command {self._command.name} completed successfully.")

    @abstractmethod
    def bake(self):
        """
        Construct the specific ASP command.

        Subclasses must implement this method to build the appropriate command
        with all necessary arguments and options.

        This method should set the `_command` attribute with an instance of the `Command` class.
        """
        pass

    @classmethod
    def from_command(cls, command: Command | str):
        """
        Create an instance of the ASPFunctionBase class from a Command object.

        Args:
            command (Command): The Command object to convert to an ASPFunctionBase instance.

        Returns:
            ASPFunctionBase: An instance of the ASPFunctionBase class.
        """
        instance = cls()

        if isinstance(command, str):
            command = Command(command)

        instance._command = command
        return instance


class ParallelStereo(ASPFunctionBase):
    """
    A class to run the parallel_stereo software from the Ames Stereo Pipeline (ASP). https://stereopipeline.readthedocs.io/en/latest/tools/parallel_stereo.html


    Args:
        asp_bin_dir (str): Path to the ASP binaries directory.
        verbose (bool): If True, enables verbose output for debugging purposes.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        super().__init__(asp_bin_dir, verbose)

    def bake(
        self,
        images: List[Path | str],
        cameras: List[Path | str] = None,
        output_file_prefix: str = "cor",
        dem: str | Path = None,
        **kwargs,
    ):
        """
        Construct the command to run the `parallel_stereo` ASP tool with the provided parameters.

        This method builds the command but does not execute it. To run the command, call the instance.

        Args:
            images (List[Path or str]):
                Input image files. If using mapprojected images, a DEM must be provided.
            cameras (List[Path or str], optional):
                Camera files. Can be None for certain image types like ISIS (see documentation for details).
                If cameras are required for your images and not provided, a warning will be issued.
            output_file_prefix (str, optional):
                The prefix for output files. It can either be a single string or a path with a subdirectory and prefix (e.g., 'output/corr'). Defaults to "cor".
            dem (str or Path, optional):
                DEM file required if using mapprojected images. Omit for non-mapprojected images.
            **kwargs:
                Any other `parallel_stereo` command-line options can be passed as keyword arguments.

        Example:
            ```python
            from pathlib import Path

            asp = ParallelStereo("/path/to/asp/bin", verbose=True)
            asp.bake(
                images=[Path("left_map_proj.tif"), Path("right_map_proj.tif")],
                cameras=None,
                dem=Path("dem_file.tif"),
                output_file_prefix="output/corr",
                job_size_w=2048,
                job_size_h=2048
            )
            asp()  # Executes the constructed parallel_stereo command
            ```
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
        command.extend(**kwargs_to_asp(kwargs))

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

        self._command = command


class BundleAdjust(ASPFunctionBase):
    """
    A class to run the bundle_adjust software from the Ames Stereo Pipeline (ASP). https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html

    Args:
        asp_bin_dir (str): Path to the ASP binaries directory.
        verbose (bool): If True, enables verbose output for debugging purposes.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initialize the BundleAdjust class.

        Args:
            asp_bin_dir (str): Path to the ASP binaries directory.
            verbose (bool): If True, enables verbose output for debugging purposes.
        """
        super().__init__(asp_bin_dir, verbose)

    def bake(
        self,
        images: List[Path | str],
        cameras: List[Path | str],
        output_prefix: str = "ba_run",
        **kwargs,
    ):
        """
        Run the ASP bundle_adjust command with the provided parameters.

        Hardcoded Parameters:
        - `images` and `cameras` are required positional arguments.
        - The output prefix defaults to "ba_run".

        Args:
            images (list of Path or str): List of input images.
            cameras (list of Path or str): List of input camera files associated with the images.
            output_prefix (str, optional): The prefix for output files. Defaults to "ba_run".
            **kwargs: Additional parameters to pass to the bundle_adjust command.

        Raises:
            FileNotFoundError: If any of the input image or camera files are missing.
            ValueError: If the number of images doesn't match the number of cameras.
        """

        # Ensure all image files exist
        for image in images:
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")

        # Ensure all camera files exist
        for camera in cameras:
            if not Path(camera).exists():
                raise FileNotFoundError(f"Camera file not found: {camera}")

        # Ensure the number of camera files matches the number of images
        if len(cameras) != len(images):
            raise ValueError(
                "The number of camera files must match the number of images."
            )

        # Initialize the Command object with the base command
        command = Command(
            cmd="bundle_adjust", name="bundle_adjust", verbose=self._verbose
        )

        # Add image files
        command.extend(images)

        # Add camera files
        command.extend(cameras)

        # Add the output file prefix
        command.extend(**{"-o": output_prefix})

        # Add any additional optional arguments from kwargs
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command


class AddSpotRPC(ASPFunctionBase):
    """
    A class to run the AddSpotRPC function https://stereopipeline.readthedocs.io/en/latest/tools/add_spot_rpc.html


    Args:
        asp_bin_dir (str): Path to the ASP binaries directory.
        verbose (bool): If True, enables verbose output for debugging purposes.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        super().__init__(asp_bin_dir, verbose)

    def bake(
        self,
        input_metadata_file: str | Path,
        **kwargs,
    ):
        """
        Construct the command to run the `add_spot_rpc` ASP tool with the provided parameters.

        This method builds the command but does not execute it. To run the command, call the instance.

        Args:
            input_metadata_file (str or Path):
                Path to the input SPOT5 metadata file.
            **kwargs:
                Any other `add_spot_rpc` command-line options can be passed as keyword arguments.

        Example:
            ```python
            asp = AddSpotRPC("/path/to/asp/bin", verbose=True)
            asp.bake(
                input_metadata_file="spot5_metadata.txt",
                min_height=100,
                max_height=5000,
            )
            asp()  # Executes the constructed add_spot_rpc command
            ```
        """
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

        # Due to a bug in ASP, the output file must be specified with the -o flag
        if "output_prefix" in kwargs:
            kwargs["o"] = kwargs.pop("output_prefix")

        # Add additional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command


class MapProject(ASPFunctionBase):
    """
    A class to run the mapproject function from the Ames Stereo Pipeline (ASP). This class is used for projecting images onto a DEM using camera models. https://stereopipeline.readthedocs.io/en/latest/tools/mapproject.html


    Args:
        asp_bin_dir (str): Path to the ASP binaries directory.
        verbose (bool): If True, enables verbose output for debugging purposes.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initialize the MapProject class.

        Args:
            asp_bin_dir (str): Path to the ASP binaries directory.
            verbose (bool): If True, enables verbose output for debugging purposes.
        """
        super().__init__(asp_bin_dir, verbose)

    def bake(
        self,
        dem: Path | str,
        camera_image: Path | str,
        camera_model: Path | str,
        output_image: Path | str,
        **kwargs,
    ):
        """
        Run the ASP mapproject command with the provided parameters.

        Args:
            dem (Path or str): The DEM file used for map projection.
            camera_image (Path or str): The camera image to be projected.
            camera_model (Path or str): The camera model associated with the camera image.
            output_image (Path or str): The output image file.
            **kwargs: Additional parameters to pass to the mapproject command.

        Raises:
            FileNotFoundError: If any of the required files (DEM, camera image, camera model) are missing.
        """

        # Ensure all input files exist
        if not Path(dem).exists():
            raise FileNotFoundError(f"DEM file not found: {dem}")

        if not Path(camera_image).exists():
            raise FileNotFoundError(f"Camera image file not found: {camera_image}")

        if not Path(camera_model).exists():
            raise FileNotFoundError(f"Camera model file not found: {camera_model}")

        # Initialize the Command object with the base command
        command = Command(cmd="mapproject", name="mapproject", verbose=self._verbose)

        # Add positional arguments (DEM, camera-image, camera-model, output-image)
        command.extend([dem, camera_image, camera_model, output_image])

        # Add optional keyword arguments from kwargs
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command


class Point2DEM(ASPFunctionBase):
    """
    A class to represent the Point2DEM function in the Ames Stereo Pipeline. https://stereopipeline.readthedocs.io/en/latest/tools/point2dem.html

    Attributes:
        asp_bin_dir (str | Path): The directory where the ASP binaries are located.
        verbose (bool): If True, provide detailed output during execution.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initializes a Point2DEM instance.

        Args:
            asp_bin_dir (str | Path, optional): The directory of the ASP binaries. Default is None.
            verbose (bool, optional): If True, provide detailed output during execution. Default is False.
        """
        super().__init__(asp_bin_dir, verbose)

    def bake(self, input_file: str | Path, **kwargs):
        """
        Executes the Point2DEM command with the specified input file and additional parameters.

        Args:
            input_file (str | Path): The input file for the Point2DEM command.
            **kwargs: Additional keyword arguments to be passed as parameters.

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        # Ensure the input file exists
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Initialize the Command object with the base command
        command = Command(cmd="point2dem", name="point2dem", verbose=self._verbose)

        # Add the input file
        command.extend(input_file)

        # Add optional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command


class Point2Las(ASPFunctionBase):
    """
    A class to represent the point2las function in the Ames Stereo Pipeline.

    Attributes:
        asp_bin_dir (str | Path): The directory where the ASP binaries are located.
        verbose (bool): If True, provide detailed output during execution.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initializes a Point2Las instance.

        Args:
            asp_bin_dir (str | Path, optional): The directory of the ASP binaries. Default is None.
            verbose (bool, optional): If True, provide detailed output during execution. Default is False.
        """
        super().__init__(asp_bin_dir, verbose)

    def bake(self, input_file: str | Path, **kwargs):
        """
        Executes the point2las command with the specified input file and additional parameters.

        Args:
            input_file (str | Path): The input PC file for the point2las command.
            **kwargs: Additional keyword arguments to be passed as parameters.

        Raises:
            FileNotFoundError: If the input PC file does not exist.
        """
        # Ensure the input PC file exists
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Initialize the Command object with the base command
        command = Command(cmd="point2las", name="point2las", verbose=self._verbose)

        # Add the input file
        command.extend(input_file)

        # Add optional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command

    def run(self):
        """
        Runs the point2las command.

        Executes the command using the run_command function with the defined parameters.
        """
        self._command.run()


class DEMGeoid(ASPFunctionBase):
    """
    A class to represent the dem_geoid function in the Ames Stereo Pipeline. https://stereopipeline.readthedocs.io/en/latest/tools/dem_geoid.html

    Attributes:
        asp_bin_dir (str | Path): The directory where the ASP binaries are located.
        verbose (bool): If True, provide detailed output during execution.
    """

    def __init__(self, asp_bin_dir: str | Path = None, verbose: bool = False):
        """
        Initializes a DEMGeoid instance.

        Args:
            asp_bin_dir (str | Path, optional): The directory of the ASP binaries. Default is None.
            verbose (bool, optional): If True, provide detailed output during execution. Default is False.
        """
        super().__init__(asp_bin_dir, verbose)

    def bake(self, input_dem: str | Path, **kwargs):
        """
        Executes the dem_geoid command with the specified input DEM and additional parameters.

        Args:
            input_dem (str | Path): The input DEM for the dem_geoid command.
            **kwargs: Additional keyword arguments to be passed as parameters.

        Raises:
            FileNotFoundError: If the input DEM does not exist.
        """
        # Ensure the input DEM exists
        if not Path(input_dem).exists():
            raise FileNotFoundError(f"Input DEM file not found: {input_dem}")

        # Initialize the Command object with the base command
        command = Command(cmd="dem_geoid", name="dem_geoid", verbose=self._verbose)

        # Add the input DEM
        command.extend(input_dem)

        # Add optional keyword arguments
        command.extend(**kwargs_to_asp(kwargs))

        self._command = command

    def run(self):
        """
        Runs the dem_geoid command.

        Executes the command using the run_command function with the defined parameters.
        """
        self._command.run()


if __name__ == "__main__":
    add_directory_to_path(
        Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
    )

    # Run parallel_stereo command

    images = ["demo/hrs-1.TIF", "demo/hrs-2.TIF"]
    metadata = ["demo/hrs-1.DIM", "demo/hrs-2.DIM"]
    seed_dem = None  # "demo/seed_dem.tif"

    # # Add Spot RPC
    # asr = AddSpotRPC()
    # asr.bake(
    #     input_metadata_file="demo/hrs-1.DIM",
    #     output_prefix="demo/hrs-1.txt",  # or o="demo/hrs-1.txt"
    #     min_height=100,
    #     max_height=4000,
    # )
    # asr()

    # # Parallel Stereo
    # ps = ParallelStereo()
    # ps.bake(
    #     images=images,
    #     cameras=metadata,
    #     output_file_prefix="output/corr",
    #     dem=seed_dem,
    #     t="spot5",
    #     stereo_algorithm="asp_bm",
    # )
    # ps()

    p2d = Point2DEM()
    p2d.bake("demo/corr-PC.tif", o="output/corr-DEM.tif", r="earth", tr=10)
    p2d()
