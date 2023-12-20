#! /usr/bin/env python
from obj_maker import create_obj_files
import pytest
import re
import os

# Update input(s) and output path below as needed

input_path = ('input_dir', [
    "/qnap/3D_stitched_LS/",
    "/qnap2/3D_stitched_LS/"
])

OUTPUT_DIRECTORY = "/qnap/Seita/output_objs/"


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

        def create_output_folders(input_dir, output_dir):
            no_npzs_folders = []
            for root, dirs, files in os.walk(input_dir):
                npz_files = [f for f in files if f.endswith('.npz') and re.search(r'\d', f) and 'structure' in f.lower()]
                relative_path = os.path.relpath(root, input_dir)

                output_path = os.path.join(output_dir, relative_path)
                
                if npz_files:

                    os.makedirs(output_path, exist_ok=True)
                    output_path = clean_output_directory(output_path)

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

                    print(f"Successfull test for {output_path}")
                else:
                    if "registration/" in relative_path.lower():
                        empty_dirs = {
                            "input": relative_path,
                            "output": output_path
                        }
                        no_npzs_folders.append(empty_dirs)

            print(f"No npz files found for ....................")        
            for directory in no_npzs_folders:
                print(f"Source Path: {directory['input']}")
                # print(f"Destination Path: {directory['output']}")


        create_output_folders(input_dir, OUTPUT_DIRECTORY)