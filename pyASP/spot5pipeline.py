from pathlib import Path

from asp import AmesStereoPipelineBase, AmesStereoPipelineError


class Spot5Pipeline(AmesStereoPipelineBase):
    def __init__(self, config_path: Path, asp_path: Path = None):
        super().__init__(config_path)

    def process_pair(self, pair_name: str, front_dir: Path, back_dir: Path):
        """Process a single Spot5 stereo pair."""
        self.logger.info(f"Processing pair: {pair_name}")

        # Define
        front_scene = front_dir / "SCENE01"
        back_scene = back_dir / "SCENE01"

        # Define metadata and imagery paths
        front_metadata = front_scene / "METADATA.DIM"
        back_metadata = back_scene / "METADATA.DIM"
        # front_imagery = front_scene / "IMAGERY.TIF"
        # back_imagery = back_scene / "IMAGERY.TIF"

        # Add RPC to metadata
        self.add_spot_rpc(front_metadata, front_metadata)
        self.add_spot_rpc(back_metadata, back_metadata)

        # Create symlinks
        self.create_symlinks(front_scene, back_scene)

        # Run Bundle Adjustment
        ba_output_dir = self.output_dir / "ba_run"
        ba_output_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_adjust(front_scene, back_scene, ba_output_dir)

        # Orthorectify Images with Bundle Adjustment Prefix
        front_map_proj_ba = front_scene / "front_map_proj_ba.tif"
        back_map_proj_ba = back_scene / "back_map_proj_ba.tif"
        self.mapproject_ba(
            front_map_proj_ba, front_scene, ba_output_dir, prefix="front"
        )
        self.mapproject_ba(back_map_proj_ba, back_scene, ba_output_dir, prefix="back")

        # Run Stereo Matching
        tmp_out = self.output_dir / "tmp" / "out"
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        self.parallel_stereo(
            front_map_proj_ba,
            back_map_proj_ba,
            front_metadata,
            back_metadata,
            tmp_out,
            self.seed_dem_path,
        )

        self.logger.info(f"Finished processing pair: {pair_name}")

    def process_all_pairs(self):
        pass

    def add_spot_rpc(self, metadata_dim: Path, output_dim: Path = None, **kwargs):
        """Add RPC metadata to a scene with the add_spot_rpc tool.
        https://stereopipeline.readthedocs.io/en/latest/tools/add_spot_rpc.html

        Args:
            metadata_dim (Path): Path to the DIM metadata file.
            output_dim (Path): Path to the output DIM file. If the output file does not exist, a new file is created containing the RPC model. Otherwise the RPC model is appended to an existing file.
            **kwargs: Additional command options.

        """
        command = ["add_spot_rpc", str(metadata_dim)]

        if output_dim is not None:
            command.extend(str(output_dim))

        command.extend(self.get_command_options(kwargs))

        self.run_command(command)

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

    def bundle_adjust(self, front_scene: Path, back_scene: Path, ba_output_dir: Path):
        """Run Bundle Adjustment."""
        command = [
            "bundle_adjust",
            "-t",
            "spot5",
            str(front_scene / "IMAGERY_FRONT.TIF"),
            str(back_scene / "IMAGERY_BACK.TIF"),
            str(front_scene / "METADATA_FRONT.DIM"),
            str(back_scene / "METADATA_BACK.DIM"),
            "-o",
            str(ba_output_dir / "out"),
            "--elevation-limit",
            str(self.config.min_z),
            str(self.config.max_z),
            "--ip-per-tile",
            "500",
            "--matches-per-tile",
            "100",
            "--threads",
            "16",
        ]
        self.run_command(command, cwd=ba_output_dir)

    def mapproject_ba(
        self, output_proj: Path, scene: Path, ba_output_dir: Path, prefix: str
    ):
        """Orthorectify images with bundle adjustment prefix."""
        command = [
            "mapproject",
            "-t",
            "rpc",
            "--t_srs",
            self.config.proj4,
            "--mpp",
            str(self.config.mapproj_in_res),
            str(self.seed_dem_path),
            str(scene / f"IMAGERY_{prefix.upper()}.TIF"),
            str(scene / f"METADATA_{prefix.upper()}.DIM"),
            str(output_proj),
            "--bundle-adjust-prefix",
            str(ba_output_dir / "out"),
        ]
        self.run_command(command, cwd=scene)

    def parallel_stereo(
        self,
        front_map_proj_ba: Path,
        back_map_proj_ba: Path,
        front_metadata: Path,
        back_metadata: Path,
        tmp_out: Path,
        seed_dem: Path,
    ):
        """Run stereo matching."""
        command = [
            "parallel_stereo",
            *self.config.pllpar.split(),
            str(front_map_proj_ba),
            str(back_map_proj_ba),
            str(front_metadata),
            str(back_metadata),
            str(tmp_out),
            str(seed_dem),
        ]
        self.run_command(command, cwd=tmp_out.parent)


if __name__ == "__main__":
    config_path = Path("config.yaml")
    pipeline = Spot5Pipeline(config_path)

    a = pipeline.run_command(["ls"])

    print("Done")
