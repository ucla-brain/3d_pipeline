#! /bin/bash

# Uses test image and parameters for testing slurm_3d_downsample

file=/panfs/dong/Luis/testing/3d_pipe_test/test_data/Ex_642_Em_680_tif_deconvoluted_3D_gaussian_6x_1000z_8bit.ims
channel=0
type=ims
resolution=50.0
base_outdir=/panfs/dong/Luis/testing/3d_test_downsample_temp_output

arg_hash=$(echo "$file $channel $type $resolution" | md5sum | awk '{print substr($0,0,6)}')
outdir="$base_outdir/temp_dir_$arg_hash"

# sbatch slurm_3d_downsample.sh $file $type $outdir $resolution $channel 
echo "downsample.py $file $type $outdir -res $resolution -c $channel"