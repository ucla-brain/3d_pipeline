#! /bin/bash

# Builds and rebuilds singularity containers for 3D pipeline
# Assumes that a remote endpoint has been set, example https://cloud.sylabs.io/builder

container=$1
processed='false'

# checks if previous command finished successfully
check_success() {
    if [ $? -ne 0 ]; then
        # echo message if provided
        echo $1
        exit 61
    fi
}

module load singularity
check_success "Unable to load Singularity using 'module load singularity'"

# Check if specific container was specified
if [ -z "$container" ]; then
    echo "No container specified (downsample, registration, transformation), removing and rebuilding all containers"
    echo "Continue (y/n)? "
    read answer
    if [ $answer != 'y' ]; then
        echo "Build canceled."
        exit 62
    fi
    container="all"
fi

echo "Specified $container container(s)"

# Build if all or downsample specified
if [ "$container" = "all" ] || [ "$container" = "downsample" ]; then
    echo "Building downsample container"
    singularity build --remote 3d_downsampling.sif singularity_ds
    check_success "Downsampling build unsuccessful"
    processed='true'
fi

# Build if all or registration specified
if [ "$container" = "all" ] || [ "$container" = "registration" ]; then
    echo "Building registration container"
    singularity build --remote 3d_registration.sif singularity_reg
    check_success "Registration build unsuccessful"
    processed='true'
fi

# Build if all or transformation specified
if [ "$container" = "all" ] || [ "$container" = "transformation" ]; then
    echo "Building transformation container"
    singularity build --remote 3d_transformation.sif singularity_trans
    check_success "transformation build unsuccessful"
    processed='true'
fi

echo Done
