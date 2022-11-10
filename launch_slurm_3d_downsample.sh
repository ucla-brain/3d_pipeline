#! /bin/bash

file=$1
channel=$2
type=$3
resolution=$4

# launch
sbatch slurm_3d_downsample.sh $file $channel $type $resolution