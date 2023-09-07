#! /usr/bin/env python
import os
import argparse
import trimesh
import numpy as np
import re
import sys


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


def create_obj_files(input_folder, output_folder, file_list, scale, translation=None, verbose=False):

    for file_name in file_list:
        input_file = os.path.join(input_folder, file_name)
        structure_num = re.findall('\d+',file_name)
        struct_num_str = int(''.join(str(x) for x in structure_num))
        output_filename = str(struct_num_str)+'.obj'
        output_file = os.path.join(output_folder, output_filename)

        verts, faces = read_npz_data(input_file)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # mesh.vertices *= scale
        # mesh.apply_transform(trimesh.transform_points(mesh,))
        if translation:
            mesh.vertices -= translation

        # Save the transformed OBJ file
        mesh.export(output_file)
        if verbose:
            print(f"Created OBJ file {output_file}")


def get_translation(fname):
    data = np.load(fname,allow_pickle=True)
    return data['xI'][0][0],data['xI'][1][0],data['xI'][2][0]


def read_npz_data(fname):
    data = np.load(fname)
    verts = [ [x, y, z] for [z, y, x] in data['verts']]
    faces = data['faces']
    return verts, faces


def main():
    parser = argparse.ArgumentParser(description='Transform OBJ files.')
    parser.add_argument('-i', '--input', help='Input path', required=True)
    parser.add_argument('-o', '--output', help='Output folder path', required=True)
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scaling factor')
    parser.add_argument('-npz', '--translation_npz', help='Translation values for (x, y, z) saved in a NPZ file.')
    parser.add_argument('-t', '--translation', default=None, help='Translation values in "x,y,z"')

    args = parser.parse_args()
    translation = None

    # Validations
    if os.path.isdir(args.input):
        npz_files = [file_name for file_name in os.listdir(args.input) if file_name.endswith('.npz')]
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
        translation = [int(val) for val in (args.translation).split(',')]
        assert len(translation) == 3, "Translation values should be in 'z, y, x' format"
    elif args.translation_npz is not None:
        translation = get_translation(args.translation_npz)

    if translation is not None:
        print(f"Using translation {translation}")

    # Create output folder if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Create object files
    print(f"Creating {len(npz_files)} OBJ files")
    create_obj_files(input_folder, args.output, npz_files, args.scale, translation)
    print("Done")

if __name__ == '__main__':    
    main()
