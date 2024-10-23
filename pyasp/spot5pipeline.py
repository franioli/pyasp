from dataclasses import dataclass
from pathlib import Path
from typing import List

from pyproj import crs

from pyasp import Command, logger
from pyasp.asp import (
    AddSpotRPC,
    AspStepBase,
    BundleAdjust,
    DEMGeoid,
    MapProject,
    ParallelStereo,
    Point2dem,
)
from pyasp.pipeline import AmesStereoPipelineBase, AmesStereoPipelineError


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


@dataclass
class Spot5PipelineConfig:
    name: str
    paths: dict[str, Path]
    pre_process: dict
    stereo: dict
    post_process: dict
    processing: dict


class Spot5Pipeline(AmesStereoPipelineBase):
    _sensor = "spot5"
    _front_scene: Path
    _back_scene: Path
    _dem_path: Path
    _out_dir: Path

    _pre_defined_default: Spot5PipelineConfig = Spot5PipelineConfig(
        name=None,
        paths={
            "front_scene": None,  # Front scene directory
            "back_scene": None,  # Back scene directory
            "output_dir": None,  # Output directory
            "dem_path": None,  # Seed DEM path
        },
        pre_process={},  # Pre-processing options
        stereo={},  # Stereo options
        post_process={},  # Post-processing options
        processing={},  # Post-processing options
    )

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
        cls._front_scene = Path(config_dict["paths"]["front_scene"])
        cls._back_scene = Path(config_dict["paths"]["back_scene"])
        cls._dem_path = Path(config_dict["paths"]["dem_path"])
        cls._out_dir = Path(config_dict["paths"]["output_dir"])

        # Make sure the output directory exists
        cls._out_dir.mkdir(parents=True, exist_ok=True)

        # Build the pipeline
        if config_dict["pre_process"]["do_compute_rpc"]:
            cls._pipeline.append(
                AddSpotRPC(
                    input_metadata_file=cls._front_scene / "METADATA.DIM",
                    min_height=config_dict["pre_process"]["min_height"],
                    max_height=config_dict["pre_process"]["max_height"],
                )
            )
            cls._pipeline.append(
                AddSpotRPC(
                    input_metadata_file=cls._back_scene / "METADATA.DIM",
                    min_height=config_dict["pre_process"]["min_height"],
                    max_height=config_dict["pre_process"]["max_height"],
                )
            )

        # Create symbolic links for front and back imagery and metadata
        create_spot5_symlinks(front_scene=cls._front_scene, back_scene=cls._back_scene)

        if config_dict["pre_process"]["do_bundle_adjust"]:
            ba_prefix = cls._out_dir / "ba_run"
            cls._pipeline.append(
                BundleAdjust(
                    images=[
                        cls._front_scene / "IMAGERY_FRONT.TIF",
                        cls._back_scene / "IMAGERY_BACK.TIF",
                    ],
                    cameras=[
                        cls._front_scene / "METADATA_FRONT.DIM",
                        cls._back_scene / "METADATA_BACK.DIM",
                    ],
                    output_prefix=ba_prefix,
                    t="spot5",
                    elevation_limit=[
                        config_dict["pre_process"]["min_height"],
                        config_dict["pre_process"]["max_height"],
                    ],
                )
            )
        else:
            ba_prefix = None

        if config_dict["pre_process"]["do_mapproject"]:
            proj4 = crs.CRS.from_epsg(config_dict["pre_process"]["epsg"]).to_proj4()

            cls._pipeline.append(
                MapProject(
                    dem=cls._dem_path,
                    camera_image=cls._front_scene / "IMAGERY_FRONT.TIF",
                    camera_model=cls._front_scene / "METADATA_FRONT.DIM",
                    output_image=cls._out_dir / "IMAGERY_FRONT_MapProj.tif",
                    t="rpc",
                    tr=config_dict["pre_process"]["mapproj_resolution"],
                    t_srs=proj4,
                    bundle_adjust_prefix=ba_prefix,
                )
            )
            cls._pipeline.append(
                MapProject(
                    dem=cls._dem_path,
                    camera_image=cls._back_scene / "IMAGERY_BACK.TIF",
                    camera_model=cls._back_scene / "METADATA_BACK.DIM",
                    output_image=cls._out_dir / "IMAGERY_BACK_MapProj.tif",
                    t="rpc",
                    tr=config_dict["pre_process"]["mapproj_resolution"],
                    t_srs=proj4,
                    bundle_adjust_prefix=ba_prefix,
                )
            )
            cls._pipeline.append(
                ParallelStereo(
                    images=[
                        cls._out_dir / "IMAGERY_FRONT_MapProj.tif",
                        cls._out_dir / "IMAGERY_BACK_MapProj.tif",
                    ],
                    cameras=[
                        cls._front_scene / "METADATA_FRONT.DIM",
                        cls._back_scene / "METADATA_BACK.DIM",
                    ],
                    output_file_prefix=cls._out_dir / "corr",
                    dem=cls._dem_path,
                    t="spot5maprpc",
                    stereo_algorithm=config_dict["stereo"]["stereo_algorithm"],
                    cost_mode=config_dict["stereo"]["cost_mode"],
                    corr_kernel=config_dict["stereo"]["corr_kernel"],
                    subpixel_mode=config_dict["stereo"]["subpixel-mode"],
                    alignment_method="NONE",
                )
            )

        cls._pipeline.append(
            Point2dem(
                input_file=cls._out_dir / "corr-PC.tif",
                r="earth",
                tr=config_dict["post_process"]["dem_resolution"],
            )
        )
        # Copy the DEM to the output directory
        out_dem_name = cls._back_scene.parent.stem.split("_HRS")[0]
        out_dem_path = (
            cls._out_dir
            / f"{out_dem_name}_{config_dict['pre_process']['epsg']}_{config_dict['post_process']['dem_resolution']}m.tif"
        )
        cls._pipeline.append(
            Command(
                f"cp {cls._out_dir}/corr-DEM.tif {out_dem_path}",
            )
        )

        if config_dict["post_process"]["geoid"] != -1:
            dem_geoid_path = (
                out_dem_path.parent
                / f"{out_dem_name}_{config_dict['post_process']['geoid']}.tif"
            )
            cls._pipeline.append(
                DEMGeoid(
                    input_dem=out_dem_path,
                    o=dem_geoid_path,
                    geoid=config_dict["post_process"]["geoid"],
                )
            )

        return cls(steps=cls._pipeline)

    def validate_config_dict(config_dict: dict) -> bool:
        # Check if the config_dict has all the required keys
        required_keys = ["paths", "pre_process", "stereo", "post_process"]
        if not all(key in config_dict for key in required_keys):
            return False

        # Check if the paths dictionary has all the required keys
        required_paths_keys = ["front_scene", "back_scene", "output_dir", "dem_path"]
        if not all(key in config_dict["paths"] for key in required_paths_keys):
            return False

        # Check if the pre_process dictionary has all the required keys
        required_pre_process_keys = [
            "do_compute_rpc",
            "do_bundle_adjust",
            "do_mapproject",
        ]
        if not all(
            key in config_dict["pre_process"] for key in required_pre_process_keys
        ):
            return False

        # Check if the stereo dictionary has all the required keys
        required_stereo_keys = [
            "stereo_algorithm",
            "cost_mode",
            "corr_kernel",
            "subpixel-mode",
            "dem_resolution",
        ]
        if not all(key in config_dict["stereo"] for key in required_stereo_keys):
            return False


if __name__ == "__main__":
    import pyasp
    from pyasp.asp import AddSpotRPC, BundleAdjust

    # Add the Ames Stereo Pipeline binaries to the PATH
    pyasp.add_asp_binary(
        Path.home() / "StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
    )

    config_dict = {
        "name": "spot5_sgm",
        "paths": {
            "front_scene": "demo/data/img/002-006_S5_053-256-0_2005-01-04-10-34-09_HRS-1_S_DT_TT/SCENE01",
            "back_scene": "demo/data/img/002-006_S5_053-256-0_2005-01-04-10-35-40_HRS-2_S_DT_TT/SCENE01",
            "output_dir": "demo/output",
            "dem_path": "demo/data/COP-DEM_GLO-30-DGED__2023_1_32632.tif",
        },
        "pre_process": {
            "do_compute_rpc": True,
            "do_bundle_adjust": True,
            "do_mapproject": True,
            "epsg": 32632,
            "mapproj_resolution": 10,
            "min_height": 100,  # Leave as None for no limit
            "max_height": 4500,  # Leave as None for no limit
        },
        "stereo": {
            "stereo_algorithm": "asp_sgm",
            "cost_mode": 3,
            "corr_kernel": [7, 7],
            "subpixel-mode": 9,
        },
        "post_process": {
            "dem_resolution": 10,
            "geoid": "EGM2008",  # -1 for not applying geoid
        },
        "processing": {
            "threads": 16,
            "processes": 10,
            "threads_multiprocess": 8,
        },
    }

    # Create a Spot5Pipeline object from a dictionary
    pipeline = Spot5Pipeline.from_dict(config_dict)

    # Resume the pipeline from a specific step
    pipeline.resume_from_step(3)

    # Run the pipeline
    pipeline.run()

    # Manually create a Spot5Pipeline object with steps
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
        "rpc_front": AddSpotRPC(
            input_metadata_file=front_dir / "METADATA.DIM",
            min_height=100,
            max_height=4500,
        ),
        "rpc_back": AddSpotRPC(
            input_metadata_file=back_dir / "METADATA.DIM",
            min_height=100,
            max_height=4500,
        ),
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
    }

    pipeline = Spot5Pipeline(steps=steps, front_scene=front_dir, back_scene=back_dir)
    pipeline.run()
