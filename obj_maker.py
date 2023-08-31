import os
import argparse
import trimesh
import numpy as np
import re

def transform_obj_files(input_folder, output_folder, scale, translation):
    file_list = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.obj')]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)    

    for file_name in file_list:
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)

        mesh = trimesh.load_mesh(input_file)
        mesh.vertices *= scale
        mesh.vertices -= translation

        # Save the transformed OBJ file
        mesh.export(output_file)

def create_obj_files(input_folder, output_folder, scale, translation):
    file_list = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.npz')]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)    

    for file_name in file_list:
        input_file = os.path.join(input_folder, file_name)
        # output_filename = file_name.replace('npz', 'obj')
        structure_num = re.findall('\d+',file_name)
        struct_num_str = int(''.join(str(x) for x in structure_num))
        output_filename = str(struct_num_str)+'.obj'
        output_file = os.path.join(output_folder, output_filename)

        verts, faces = read_npz_data(input_file)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # mesh.vertices *= scale
        # mesh.apply_transform(trimesh.transform_points(mesh,))
        mesh.vertices -= translation
        
        # Save the transformed OBJ file
        mesh.export(output_file)


def main():
    print('Running obj_maker')
    parser = argparse.ArgumentParser(description='Transform OBJ files.')
    parser.add_argument('-i', '--input', help='Input folder path')
    parser.add_argument('-o', '--output', help='Output folder path')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scaling factor')
    # parser.add_argument('-t', '--translation', nargs=3, type=float, default=[1.0, 1.0, 1.0], help='Translation values for (x, y, z)')
    parser.add_argument('-npz', '--translation_npz', help='Translation values for (x, y, z) saved in a NPZ file.')
    parser.add_argument('-t', '--translation', default=None, help='Translation values in "x,y,z"')
    

    args = parser.parse_args()

    print('transforming')
    if not args.input or not args.output:
        parser.error('Input and output folder paths are required.')

    if args.translation is not None:
        translation = [int(val) for val in (args.translation).split(',')]
        assert len(translation) == 3, "Translation values should be in 'x,y,z' format"
    elif args.translation_npz is not None:
        translation = get_translation(args.translation_npz)

    # transform_obj_files(args.input, args.output, args.scale, translation)
    # Creating 
    print('creating objs')
    create_obj_files(args.input, args.output, args.scale, translation)

def get_translation(fname):
    data = np.load(fname,allow_pickle=True)
    return data['xI'][0][0],data['xI'][1][0],data['xI'][2][0]

def read_npz_data(fname):
    data = np.load(fname)
    #verts = data['verts']  # verts in zyx
    verts = [ [x, y, z] for [z, y, x] in data['verts']]
    faces = data['faces']
    return verts, faces

if __name__ == '__main__':    
    main()
