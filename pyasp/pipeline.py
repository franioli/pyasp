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

    def run(self):
        """Run the entire stereo pipeline."""
        for step in self._pipeline:
            logger.info(f"Running step: {step}")
            step()
            logger.info(f"Finished step: {step}. Time elapsed: {step.elapsed_time}")


def create_spot5_symlinks(front_scene: Path, back_scene: Path):
    """Create symbolic links for front and back imagery and metadata."""

    name_map = {front_scene: "front", back_scene: "back"}

    for scene in [front_scene, back_scene]:
        scene_name = name_map[scene]

        # Define the symlink paths
        imagery_link = scene / f"IMAGERY_{scene_name.upper()}.TIF"
        metadata_link = scene / f"METADATA_{scene_name.upper()}.DIM"

        # Define the target paths and ensure they're absolute
        imagery_target = (scene / "IMAGERY.TIF").resolve()
        metadata_target = (scene / "METADATA.DIM").resolve()

        # Remove existing symlinks if present
        if imagery_link.is_symlink() or imagery_link.exists():
            imagery_link.unlink()
        if metadata_link.is_symlink() or metadata_link.exists():
            metadata_link.unlink()

        try:
            # Create symbolic links using absolute paths
            imagery_link.symlink_to(imagery_target)
            metadata_link.symlink_to(metadata_target)
        except Exception as e:
            logger.error(f"Failed to create symlinks: {e}")
            raise AmesStereoPipelineError("Symlink creation failed.") from e

    logger.debug("Created symbolic links for front and back imagery and metadata.")


class Spot5Pipeline(AmesStereoPipelineBase):
    _sensor = "Spot5"

    _pre_defined = {"mapprojected_sgm": {}}

    def __init__(
        self,
        steps: List[AspStepBase] | dict[str, AspStepBase],
        front_scene: Path = None,
        back_scene: Path = None,
    ):
        if front_scene and back_scene:
            self.create_symlinks(front_scene=front_scene, back_scene=back_scene)
        super().__init__(steps=steps)

    def create_symlinks(self, front_scene: Path, back_scene: Path):
        """Create symbolic links for front and back imagery and metadata."""
        create_spot5_symlinks(front_scene=front_scene, back_scene=back_scene)

    @classmethod
    def from_predefined(cls, pipeline_name: str, **kwargs):
        pass

    @classmethod
    def from_config(cls, config_path: Path):
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        pass


if __name__ == "__main__":
    from pyasp.asp import BundleAdjust, add_directory_to_path

    add_directory_to_path(
        Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
    )

    front_dir = Path(
        "demo/data/img/002-006_S5_053-256-0_2005-01-04-10-34-09_HRS-1_S_DT_TT/SCENE01"
    )
    back_dir = Path(
        "demo/data/img/002-006_S5_053-256-0_2005-01-04-10-35-40_HRS-2_S_DT_TT/SCENE01"
    )
    seed_dem_path = Path("demo/data/COP-DEM_GLO-30-DGED__2023_1_32632.tif")
    output_dir = Path("demo/output")

    image_pair = [
        front_dir / "IMAGERY_FRONT.TIF",  # Front imagery
        back_dir / "IMAGERY_BACK.TIF",  # Back imagery
    ]
    camera_pair = [
        front_dir / "METADATA_FRONT.DIM",  # Front metadata
        back_dir / "METADATA_BACK.DIM",  # Back metadata
    ]

    steps = {
        # "rpc_front": AddSpotRPC(
        #     input_metadata_file=front_dir / "METADATA.DIM",
        #     min_height=100,
        #     max_height=4500,
        # ),
        # "rpc_back": AddSpotRPC(
        #     input_metadata_file=back_dir / "METADATA.DIM",
        #     min_height=100,
        #     max_height=4500,
        # ),
        "ba": BundleAdjust(
            images=image_pair,
            cameras=camera_pair,
            elevation_limit=[0, 4500],
            output_prefix="output/ba_run",
            t="spot5",
            ip_per_tile=500,
            matches_per_tile=100,
            threads=16,
        ),
        # "map": MapProject(),
        # "stereo": ParallelStereo(),
    }

    pipeline = Spot5Pipeline(steps=steps, front_scene=front_dir, back_scene=back_dir)
    pipeline.run()
