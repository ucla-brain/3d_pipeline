# 3D Pipeline
3D image processing pipeline

<!-- GETTING STARTED -->
## Getting Started

Repo contains code for the 3D image processing pipeline


### Prerequisites

* Clone the repo
    ```sh
    git clone git@github.com:ucla-brain/3d_pipeline.git
    cd 3d_pipeline
    ```
* If testing locally, a virtual environment can be set up with [anaconda](https://docs.conda.io/en/latest/index.html) or similar.
* For the registration step, access to a GPU and CUDA may be necessary. 
* Note that if using the launch scripts, the scripts use environment module singularity and slurm.
* If using the container build script, you'll need a [sylabs account](https://cloud.sylabs.io) to do a remote build, and to verify your account by running on bmap with
    ```
    singularity remote login
    ```
* [twardlab emlddmm repo](https://github.com/twardlab/emlddmm.git) is required for registration step. Specifically, the emlddmm.py script 


### Local Installation

1. Create virtual environment, with conda as an example
   ```sh
   conda create --name 3d_pipeline
   ```
2. Activate conda environment
   ```sh
   conda activate 3d_pipeline
   ```
3. Install requirements from requirements.txt file
   ```sh
   pip install -r requirements.txt
   ```

### Building Singularity Containers

1. While on bmap login node, activate singularity module
   ```sh
   module load singularity
   ```
2. If first time, make sure a remote connection has been established with a remote container builder like the bmap recommended [sylabs](https://cloud.sylabs.io). You'll need to create a key on sylabs and verify it with 
   ```sh
   singularity remote login
   ```
3. To build containers, run build script and specify which container should be built (downsample, registration, transformation) or leave blank to remove and build all containers
   ```sh
   ./build_pipeline_containers.sh <container>
   ```

<!-- USAGE EXAMPLES -->
## Usage

Example local usage

```sh
python downsample.py /path/to/image.ims ims /path/to/temp_dir_4059c0 -res 50.0 -c 0 
```
```sh
python registration.py /path/to/target.npz /path/to/outdir /path/to/annotation.vtk file_name /path/to/atlas /path/to/template
```
```sh
python  obj_maker.py  --input /path/to/structure_000000000997_surface_root.npz --output /path/to/outdir  --translation='x,y,z' --rotation_matrix='x1,y1,z1,x2,y2,z3,x3,y3,z3'
```

launch script usage
```sh
launch_slurm_3d_downsample.sh $file $channel $img_type $resolution $base_outdir
```

```sh
launch_slurm_3d_registration.sh $target_name $output_dir $seg_name $output_file $atlas $template
```

<!-- Testing -->
## Tests

Setup testing environment using conda. 

Important: Python (and Pytest) must use Python version 3.6 and above.

1. Create virtual environment using conda
   ```sh
   conda env create -f environment.yml
   ```

2. Activate conda environment
   ```sh
   conda activate 3d_pipeline_pytest
   ```

3. Run tests
   ```sh
   pytest -s
   ```
