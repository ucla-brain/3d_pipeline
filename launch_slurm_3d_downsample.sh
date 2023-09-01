#! /bin/bash

file=$1
channel=$2
img_type=$3
resolution=$4
base_outdir=$5

# Create unique hash for outdir based on args, this avoids temp directory conflicts
arg_hash=$(echo "$file $channel $img_type $resolution" | md5sum | awk '{print substr($0,0,6)}')
outdir="$base_outdir/temp_dir_$arg_hash"

# launch
sbatch slurm_3d_downsample.sh $file $img_type $outdir $resolution $channel