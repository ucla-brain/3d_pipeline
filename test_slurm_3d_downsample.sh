#! /bin/bash

# Uses test image and parameters for testing slurm_3d_downsample

file=/panfs/dong/Luis/testing/3d_pipe_test/test_data/Ex_642_Em_680_tif_deconvoluted_3D_gaussian_6x_1000z_8bit.ims
channel=0
type=ims
resolution=50.0

sbatch slurm_3d_downsample.sh $file $channel $type $resolution