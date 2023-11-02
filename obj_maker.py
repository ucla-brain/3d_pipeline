#! /usr/bin/env python
import os
import argparse
import trimesh
import numpy as np
import re
import sys
from tqdm import tqdm
import time
import shutil


def transform_obj_files(input_folder, output_folder, scale, translation):
    file_list = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.obj')]

    for file_name in file_list:
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)

        mesh = trimesh.load_mesh(input_file)
        mesh.vertices *= scale
        mesh.vertices -= translation

        # Save the transformed OBJ file
        mesh.export(output_file)


def create_obj_files(input_folder, output_folder, file_list, scale, translation=None, matrix=None, verbose=False):

    start_time = time.time()
    generated_obj_files = []
    for file_name in tqdm(file_list, desc="Create OBJ files"):

        input_file = os.path.join(input_folder, file_name)

        structure_num = None
        file_name_splits = file_name.split('_')
        for part in file_name_splits:
            if part.isdigit():
                structure_num = int(part)
                break
        if structure_num == None:
            print("No match found.")
        else:
            print(structure_num)

        print(f'Rename for {file_name} using {structure_num} for conversion...')
        output_filename = str(structure_num)+'.obj'
        output_file = os.path.join(output_folder, output_filename)
        generated_obj_files.append(output_file)

        verts, faces = read_npz_data(input_file)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # mesh.vertices *= scale
        if matrix is not None:
            mesh.apply_transform(matrix)
        if translation:
            mesh.vertices -= translation

        # Save the transformed OBJ file
        mesh.export(output_file)
        if verbose:
            print(f"Created OBJ file {output_file}")

    # Check the generated obj files
    try:
        assert len(file_list) == len(generated_obj_files)
    except AssertionError:
        print(f"NPZ list has {len(file_list)} items, and OBJ list has {len(generated_obj_files)} items.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time} seconds")

# deprecated
#def get_translation(fname):
#    data = np.load(fname,allow_pickle=True)
#    return data['xI'][0][0],data['xI'][1][0],data['xI'][2][0]

def parse_transformation_matrix(matrix_str):
    matrix_values = np.array(matrix_str.split(','), dtype=float)
    transformation_matrix = matrix_values.reshape(3, 3)
    return transformation_matrix

def get_origin_offset(fname, transformation_matrix):
    data = np.load(fname,allow_pickle=True)
    origin = (data['origin'][2], data['origin'][1], data['origin'][0]) # data['origin'] is stored z,y,x
    transformed_origin = np.dot(transformation_matrix, origin)
    return list(transformed_origin.astype(int))  # Round and convert to integers

def read_npz_data(fname):
    data = np.load(fname)
    verts = [ [x, y, z] for [z, y, x] in data['verts']]
    faces = data['faces']
    return verts, faces

def create_matrix(matrix):
    input_matrix = [int(x.strip()) for x in (matrix).split(',')]
    assert len(input_matrix) == 9, "The 9 rotation values should be in 'x1,y1,z1,x2,y2,z2,x3,y3,z3' format"
    rotation_matrix = [input_matrix[i:i+3] for i in range(0, 9, 3)]

    matrix_4x4 = np.eye(4)  # Initialize a 4x4 identity matrix
    matrix_4x4[:3, :3] = rotation_matrix # Copy the input matrix the top-left corner

    return matrix_4x4


def main():
    parser = argparse.ArgumentParser(description='Transform OBJ files.')
    parser.add_argument('-i', '--input', help='Input path', required=True)
    parser.add_argument('-o', '--output', help='Output folder path', required=True)
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scaling factor')
    parser.add_argument('-npz', '--translation_npz', help='Translation values for (x, y, z) saved in a NPZ file.')
    parser.add_argument('-t', '--translation', default=None, help='Translation values in "x,y,z"')
    parser.add_argument('-r', '--rotation_matrix', default=None, help='Rotation (reorientation) matrix in x1,y1,z1,x2,y2,z2,x3,y3,z3. For [x1,y1,z1],[x2,y2,z2],[x3,y3,z3]"')

    args = parser.parse_args()
    translation = None

    # Validations
    if os.path.isdir(args.input):
        npz_files = [file_name for file_name in os.listdir(args.input) if file_name.endswith('.npz') and file_name.startswith('structure_')]
        assert len(npz_files) > 0, f"Input folder {args.input} doesn't contain npz files"
        input_folder = args.input
    elif os.path.isfile(args.input):
        assert str(args.input).endswith('.npz'), "Input file is not a valid npz file"
        input_folder, npz = os.path.split(args.input)
        npz_files = [npz]
    else:
        sys.exit(f"{args.input} is not valid input")

    # Get translation values if available
    if args.translation is not None:
        translation = [float(val) for val in (args.translation).split(',')]
        translation = [i * -1 for i in translation]
    elif args.translation_npz is not None:
        # translation = get_translation(args.translation_npz) # This is the old way of getting offset value from the downsample .npz file
        rot_mat = parse_transformation_matrix(args.rotation_matrix)
        translation = get_origin_offset(args.translation_npz, rot_mat)

    assert len(translation) == 3

    # remove the output folder first
    if os.path.exists(args.output):
        shutil.rmtree(args.output, ignore_errors=True)
    os.makedirs(args.output, exist_ok=True)    

    # Create object files
    print(f"Creating {len(npz_files)} OBJ files")
    if args.rotation_matrix is not None:
        matrix = create_matrix(args.rotation_matrix)
        create_obj_files(input_folder, args.output, npz_files, args.scale, translation, matrix)
    else:
        create_obj_files(input_folder, args.output, npz_files, args.scale, translation)
    print("Done")

if __name__ == '__main__':    
    main()
