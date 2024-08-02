#! /usr/bin/env python
from obj_maker import create_obj_files
import pytest
import re
import os
import numpy as np

# Update input(s) and output path below as needed

input_path = ('input_dir', [
    "/qnap/3D_stitched_LS/",
    "/qnap2/3D_stitched_LS/"
])

OUTPUT_DIRECTORY = "/qnap2/output_objs/"


class TestObjMaker:

    @pytest.fixture(scope="function")
    def extract_number_from_filename(self):
        def extract(filename):
            parts = os.path.splitext(filename)[0].split('_')
            for part in parts:
                if part.isdigit():
                    return int(part)
        return extract

    @pytest.fixture(scope="function")
    def count_files(request, extract_number_from_filename):
        unique_numbers = set()

        def cleanup_unique_numbers():
            unique_numbers.clear()

        def extract_file_count(directory, extension):
            count = 0
            for filename in os.listdir(directory):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path) and filename.endswith(extension):
                    number = extract_number_from_filename(filename)
                    if number is not None and number not in unique_numbers:
                        unique_numbers.add(number)
                        count += 1

            cleanup_unique_numbers()

            return count

        return extract_file_count

    @pytest.fixture(scope="function")
    def clean_output_directory(self):
        def clean_contents(output_path):
            if os.listdir(output_path):
                for item in os.listdir(output_path):
                    item_path = os.path.join(output_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        os.rmdir(item_path)
            return output_path
        return clean_contents


    @pytest.mark.parametrize(*input_path)
    def test_create_obj_files(self, input_dir, extract_number_from_filename, count_files,  clean_output_directory):

        if not os.path.exists(input_dir):
            assert False, f"Input directory does not exist, skipping test. Invalid directory: {input_dir}"

        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

        def contains_obj_files(directory):
            # List all files in the given directory
            files = os.listdir(directory)
            
            # Check if any file has the .obj extension
            for file in files:
                if file.endswith(".obj"):
                    return True
            return False

        def find_npz_file_with_keyword(directory, keyword):
            keyword_lower = keyword.lower()
            for item in os.listdir(directory):
                if item.lower().endswith('.npz') and keyword_lower in item.lower():
                    return os.path.join(directory, item)
            return None


        def create_output_folders(input_dir, output_dir):
            no_npzs_folders = []
            for root, dirs, files in os.walk(input_dir):
                npz_files = [f for f in files if f.endswith('.npz') and re.search(r'\d', f) and 'structure' in f.lower()]
                relative_path = os.path.relpath(root, input_dir)

                output_path = os.path.join(output_dir, relative_path)

                if npz_files:

                    os.makedirs(output_path, exist_ok=True)
                    output_path = clean_output_directory(output_path)

                    # if "/Archived" in output_path or "Yongsoo" in output_path:
                    if "/Archived" in output_path or "fmost" not in output_path.lower():
                        continue
                    
                    # TODO: Check if any file has the .obj extension so that we know they have been generated
                    # print(f'{output_path} ........')
                    # if os.path.exists(output_path):
                    #     if contains_obj_files(output_path):
                    #         # if "fmost" in output_path:
                    #         print(f'{output_path} contains OBJ files.....')
                    #         continue
                    # else:
                    #     print(f'{output_path} does not exist...')

                    create_obj_files(root, output_path, npz_files, 1, None, None, False)

                    npz_count = count_files(root, '.npz')
                    obj_count = count_files(output_path, '.obj')

                    assert obj_count >= npz_count, f"Lower obj count generated: obj(s): {obj_count}, npz(s): {npz_count}"
                    assert npz_count >= obj_count, f"Lower npz count error: {npz_count}, obj(s): {obj_count}"
                    assert npz_count > 0, "Empty input file used."


                    for input_file in npz_files:
                        num = extract_number_from_filename(input_file)
                        expected_output = f"{num}.obj"

                        assert os.path.exists(os.path.join(output_path, expected_output)), \
                            f"Expected output file {expected_output} not found for {input_file}."

                    root997_file_path = find_npz_file_with_keyword(root, '997')
                    if root997_file_path:
                        print(f"Found file: {root997_file_path}")
                        npz997 = np.load(root997_file_path,allow_pickle=True)
                        print(f"Successfull test for INPUT:{root}  OUTPUT:{output_path} and OFFSET:{npz997['origin']}\n\n\n")
                    else:
                        print(f"Successfull test for INPUT:{root}  OUTPUT:{output_path} and OFFSET:No 997.npz\n\n\n")
                        

                else:
                    if "registration/" in relative_path.lower():
                        empty_dirs = {
                            "input": root,
                            "output": output_path
                        }
                        no_npzs_folders.append(empty_dirs)

            #print(f"No npz files found for ....................")        
            #for directory in no_npzs_folders:
            #    path = {directory['input']}
            #    if "/QC" not in path:
            #        print(f"Source Path: {path}")
                    # print(f"Destination Path: {directory['output']}")

    # @pytest.mark.parametrize(*input_path)
    # @pytest.mark.skip(reason="Skipping test for now")    
    # def test_check_most_dirs(self, input_dir, caplog):
    #     most_dirs = []
    #     no_most_dirs = []

    #     # List all directories one level below input_dir
    #     try:
    #         subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    #     except FileNotFoundError:
    #         print(f"Directory not found: {input_dir}")
    #         return

    #     for subdir in subdirs:
    #         registration_dir = os.path.join(subdir, "Registration")
    #         if os.path.isdir(registration_dir):
    #             if "fMOST_10um" in os.listdir(registration_dir):
    #                 most_dirs.append(registration_dir)
    #             else:
    #                 no_most_dirs.append(registration_dir)

    #     if most_dirs:
    #         print(f"'fMOST' directory found in the following 'Registration' directories under {input_dir}:")
    #         for dir in most_dirs:
    #             print(f"  - {dir}")
    #     else:
    #         print(f"No 'fMOST' directory found in any 'Registration' subdirectory under {input_dir}.")

    #     if no_most_dirs:
    #         print(f"'fMOST' directory not found in the following 'Registration' directories under {input_dir}:")
    #         for dir in no_most_dirs:
    #             print(f"  - {dir}")

        create_output_folders(input_dir, OUTPUT_DIRECTORY)
        # test_check_most_dirs(input_dir)

    # Run pytest with the -s flag to see the output on the console
    # pytest -s

