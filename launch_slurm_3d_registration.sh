#! /bin/bash

target_name=$1
output_dir=$2
seg_name=$3
output_file=$4
atlas=$5
template=$6

# launch
sbatch slurm_3d_registration.sh $target_name $output_dir $seg_name $output_file $atlas $template
