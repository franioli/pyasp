#!/bin/bash


# Add ASP binary directory to the PATH
asp_bin_dir="$HOME/StereoPipeline-3.5.0-alpha-2024-10-06-x86_64-Linux/bin"
export PATH=${PATH}:${asp_bin_dir}                                   

# Paths
data_path=`realpath "data"`
seed_dem_path=`realpath "data/COP-DEM_GLO-30-DGED__2023_1_32632.tif"`
front_dir=`realpath $data_path/img/002-006_S5_053-256-0_2005-01-04-10-34-09_HRS-1_S_DT_TT/SCENE01`
back_dir=`realpath $data_path/img/002-006_S5_053-256-0_2005-01-04-10-35-40_HRS-2_S_DT_TT/SCENE01`

dir_out_path=`realpath "output_demcop30"`
dir_out_dem_path=$dir_out_path/dem
dem_name="002-006_S5_053-256-0_2005-01-04"

mkdir -p $dir_out_dem_path

# --- Coordinate system
epsg_code=32362
proj4="+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +type=crs"

mapproj_in_res=20
dem_res=20

# Approximate minimum and maximum elevation of the scene, used for add_spot_rpc.
minZ=100
maxZ=4500

# compute the RPCs and append them to the metadata files
add_spot_rpc $front_dir/METADATA.DIM --min-height $minZ --max-height $maxZ -o $front_dir/METADATA.DIM
add_spot_rpc $back_dir/METADATA.DIM --min-height $minZ --max-height $maxZ -o $back_dir/METADATA.DIM

# Run Bunndle Adjustment
#- The files cannot be passed into the bundle_adjust tool with the same file name even if they are in different folders. Create symlinks to the front and back imagery and metadata https://stereopipeline.readthedocs.io/en/latest/examples/spot5.html
ln -s  "$front_dir/METADATA.DIM" "$front_dir/METADATA_FRONT.DIM"
ln -s  $front_dir/IMAGERY.TIF $front_dir/IMAGERY_FRONT.TIF
ln -s  $back_dir/METADATA.DIM $back_dir/METADATA_BACK.DIM
ln -s  $back_dir/IMAGERY.TIF $back_dir/IMAGERY_BACK.TIF
bundle_adjust -t spot5  \
  $front_dir/IMAGERY_FRONT.TIF $back_dir/IMAGERY_BACK.TIF \
  $front_dir/METADATA_FRONT.DIM $back_dir/METADATA_BACK.DIM \
  -o $dir_out_path/ba_run/out  --elevation-limit $minZ $maxZ \
  --ip-per-tile 500 --matches-per-tile 100 \
  --threads 16 

# Orthorectify the images
mapproject -t rpc --t_srs "$proj4" --tr $mapproj_in_res $seed_dem_path $front_dir/IMAGERY_FRONT.TIF $front_dir/METADATA_FRONT.DIM $dir_out_path/front_map_proj_ba.tif --bundle-adjust-prefix $dir_out_path/ba_run/out

mapproject -t rpc --t_srs "$proj4" --tr $mapproj_in_res $seed_dem_path $back_dir/IMAGERY_BACK.TIF $back_dir/METADATA_BACK.DIM $dir_out_path/back_map_proj_ba.tif --bundle-adjust-prefix $dir_out_path/ba_run/out

# Run stereo matching
parallel_stereo -t spot5maprpc corr-kernel 8 --processes 6 --threads-multiprocess 8 $dir_out_path/front_map_proj_ba.tif $dir_out_path/back_map_proj_ba.tif $front_dir/METADATA.DIM $back_dir/METADATA.DIM $dir_out_path/corr/corr $seed_dem_path
# To resume the stereo matching from a previous run, use the following option:
# --resume-at-corr

point2dem -r earth --tr $dem_res $dir_out_path/corr/corr-PC.tif

cp $dir_out_path/corr/corr-DEM.tif $dir_out_dem_path/${dem_name}_DEM_${dem_res}m.tif
dem_geoid --geoid EGM2008 $dir_out_dem_path/${dem_name}_DEM_${dem_res}m.tif -o $dir_out_dem_path/${dem_name}_DEM_${dem_res}m_egm08.tif