#! /bin/bash

# Uses test downsampled image and parameters for testing slurm_3d_registration

target_name="/panfs/dong/Luis/repos/3d_pipeline_testing/SW220406_01_LS_6x_1000z_ch_0_pow_[0.125]_down.npz"
output_dir="/panfs/dong/Luis/testing/3d_test_registration_output/"
output_file="test_registration_output"
seg_name="/panfs/dong/Luis/testing/allen_vtk/annotation_50.vtk"
atlas="/panfs/dong/Luis/testing/allen_vtk/ara_nissl_50.vtk"
template="/panfs/dong/Luis/testing/allen_vtk/average_template_50.vtk"

cd ..
sbatch slurm_3d_registration.sh $target_name $output_dir $seg_name $output_file $atlas $template
