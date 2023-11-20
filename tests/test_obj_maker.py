#! /usr/bin/env python
from obj_maker import create_obj_files
import numpy as np
import pytest
import tempfile
import shutil
import re
import os
import sys
from io import StringIO

# Update input(s) and output path below as needed

input_path = ('input_dir', [
    "/qnap/3D_stitched_LS/"
])

@pytest.fixture
def output_directory():
    pytest.output_directory = "/qnap/ChristianE/output_objs/"



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



    @pytest.fixture
    def validate_paths(self, request, input_dir, output_directory):
        output_dir = pytest.output_directory
        if not os.path.exists(input_dir):
            assert False, f"Input directory does not exist, skipping test. Invalid directory: {input_dir}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    



    @pytest.mark.parametrize(*input_path)
    def test_create_obj_files(self, capsys, input_dir, extract_number_from_filename, count_files,  clean_output_directory, output_directory, validate_paths):

        output_dir = pytest.output_directory
        
        if not (os.path.exists(input_dir) and os.path.exists(output_dir)):
            validate_paths(input_dir, output_dir)

        def create_output_folders(input_dir, output_dir):
            for root, dirs, files in os.walk(input_dir):
                npz_files = [f for f in files if f.endswith('.npz') and re.search(r'\d', f) and 'structure' in f.lower()]
                relative_path = os.path.relpath(root, input_dir)

                if npz_files:
                    original_stdout = sys.stdout
                    fake_stdout = StringIO()
                    sys.stdout = fake_stdout

                    output_path = os.path.join(output_dir, relative_path)
                    os.makedirs(output_path, exist_ok=True)
                    output_path = clean_output_directory(output_path)
                    create_obj_files(root, output_path, npz_files, 1, None, None, False)
                    captured_output = fake_stdout.getvalue()

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

                    sys.stdout = original_stdout
                    print(f"Successfull test for {relative_path}")
                else:
                    if "registration/" in relative_path.lower():
                        print(f"No npz's found for {relative_path}")

        create_output_folders(input_dir, output_dir)
