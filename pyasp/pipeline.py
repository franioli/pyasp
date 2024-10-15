from abc import ABC
from pathlib import Path
from typing import List

from pyasp import logger
from pyasp.asp import AspStepBase


class AmesStereoPipelineError(Exception):
    """Custom exception for Ames Stereo Pipeline errors."""

    pass


class AmesStereoPipelineBase(ABC):
    _sensor = ""
    _pipeline = []

    def __init__(self, steps: List[AspStepBase] | dict[str, AspStepBase]):
        if isinstance(steps, dict):
            steps = [step for step in steps.values() if isinstance(step, AspStepBase)]

        # Check if all steps are AspStepBase objects
        for step in steps:
            if not isinstance(step, AspStepBase):
                raise TypeError(
                    f"Invalid {step} in steps. All steps must be AspStepBase objects."
                )
        self._pipeline = steps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} Pipeline with steps: {self._pipeline}"

    def run_pipeline(self):
        """Run the entire stereo pipeline."""
        for step in self._pipeline:
            logger.info(f"Running step: {step}")
            step.run()
            logger.info(f"Finished step: {step}. Time elapsed: {step.elapsed_time}")


class Spot5Pipeline(AmesStereoPipelineBase):
    _sensor = "Spot5"

    def __init__(
        self,
        steps: List[AspStepBase] | dict[str, AspStepBase],
        front_dir: Path,
        back_dir: Path,
        seed_dem_path: Path,
        output_dir: Path,
    ):
        super().__init__()

        self.create_symlinks(front_scene=front_dir, back_scene=back_dir)

    def create_symlinks(self, front_scene: Path, back_scene: Path):
        """Create symbolic links for front and back imagery and metadata."""
        try:
            # Front Symlinks
            front_metadata_link = front_scene / "METADATA_FRONT.DIM"
            front_imagery_link = front_scene / "IMAGERY_FRONT.TIF"
            if not front_metadata_link.exists():
                front_metadata_link.symlink_to(front_scene / "METADATA.DIM")
            if not front_imagery_link.exists():
                front_imagery_link.symlink_to(front_scene / "IMAGERY.TIF")

            # Back Symlinks
            back_metadata_link = back_scene / "METADATA_BACK.DIM"
            back_imagery_link = back_scene / "IMAGERY_BACK.TIF"
            if not back_metadata_link.exists():
                back_metadata_link.symlink_to(back_scene / "METADATA.DIM")
            if not back_imagery_link.exists():
                back_imagery_link.symlink_to(back_scene / "IMAGERY.TIF")

            self.logger.debug(
                "Created symbolic links for front and back imagery and metadata."
            )
        except Exception as e:
            self.logger.error(f"Failed to create symlinks: {e}")
            raise AmesStereoPipelineError("Symlink creation failed.") from e


if __name__ == "__main__":
    from pyasp.asp import *

    add_directory_to_path(
        Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
    )

    front_dir = Path(
        "demo/002-006_S5_053-256-0_2005-01-04-10-34-09_HRS-1_S_DT_TT/SCENE01"
    )
    back_dir = Path(
        "demo/002-006_S5_053-256-0_2005-01-04-10-35-40_HRS-2_S_DT_TT/SCENE01"
    )
    seed_dem_path = Path("demo/COP-DEM_GLO-30-DGED__2023_1_32632.tif")
    output_dir = Path("demo/output")

    steps = {
        "rpc_front": AddSpotRPC(input_metadata_file=front_dir / "METADATA.DIM"),
        "rpc_back": AddSpotRPC(input_metadata_file=back_dir / "METADATA.DIM"),
        "ba": BundleAdjust(),
        "map": MapProject(),
        "stereo": ParallelStereo(),
    }

    pipeline = Spot5Pipeline(steps)
