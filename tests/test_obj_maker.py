#! /usr/bin/env python
from obj_maker import create_obj_files
import numpy as np
import pytest
import tempfile
import shutil
import re
import os

# Update input paths below for individual function tests (small folder size recommended)
test_paths = [
    # "/ifshome/cestrada/3d_pipeline/input/",
]

# Update input and output paths for testing existing obj paths
completed_paths = ('input_dir, output_dir', [
# B1
    ("/qnap/3D_stitched_LS/20220605_SW220406_01_LS_6x_1000z_B1/Registration/Yongsoo_20um_v2/", "/qnap/ChristianE/output_objs/B1/"),
# B2
    ("/qnap/3D_stitched_LS/20220725_SW220510_02_LS_6x_1000z_B2/Registration/Yangsoo_20um/","/qnap/ChristianE/output_objs/B2/"),
# B3
    ("/qnap/3D_stitched_LS/20230527_SM230223_04_LS_15x_800z_Double_MORF_B3/Registration/Yongsoo_20um/","/qnap/ChristianE/output_objs/B3/") ,
# B4
    ("/qnap/3D_stitched_LS/20230624_SM230223_05_LS_15x_800z_Double_MORF_B4/Registration/Yongsoo_20um/","/qnap/ChristianE/output_objs/B4/") ,
# B5
    ("/qnap/3D_stitched_LS/20230724_SM230601_05_LS_15x_800z_B5/Registration/Yongsoo_20um/","/qnap/ChristianE/output_objs/B5/") ,
# B6
    ("/qnap/3D_stitched_LS/20230825_SM230601_06_LS_15x_800z_B6/Registration/Yongsoo_20um/","/qnap/ChristianE/output_objs/B6/"),
# NB1
    ("/qnap/3D_stitched_LS/20220818_SW220405_05_LS_6x_1000z_NB1/Registration/Yongsoo_20um/structures_v1/","/qnap/ChristianE/output_objs/NB1/") ,
# NB2
    ("/qnap/3D_stitched_LS/20230419_SM221011_06_LS_15x_800z_NB2/Registration/Yongsoo_20um/","/qnap/ChristianE/output_objs/NB2/")  ,
])



class TestObjMaker:


    @pytest.fixture(scope="function")
    def temp_output_dir(self, request):
        temp_dir = tempfile.mkdtemp(prefix="tmp_obj_")
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="function")
    def extract_number_from_filename(self):
        def extract(filename):
            parts = os.path.splitext(filename)[0].split('_')
            for part in parts:
                if part.isdigit():
                    return int(part)
        return extract

    @pytest.fixture(scope="function")
    def count_files(self):
        def extract_file_count(directory, extension):
            count = 0
            for filename in os.listdir(directory):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path) and filename.endswith(extension) and re.search(r'\d', filename):
                    count += 1
            return count
        return extract_file_count
    
    ## Internal tests ##


    # Testing new obj files generated (both counts and names)
    @pytest.mark.parametrize("input_dir", test_paths)
    def test_create_obj_files(self, input_dir, temp_output_dir, extract_number_from_filename, count_files):
        if not (os.path.exists(input_dir)):
            assert False, f"Input directory does not exist, skipping test. Directory: {input_dir}"
        file_list = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.npz')]
        output_dir = temp_output_dir
        create_obj_files(input_dir, output_dir, file_list, 1, None, None, False)

        npz_count = count_files(input_dir, '.npz')
        obj_count = count_files(output_dir, '.obj')

        assert npz_count == obj_count

        for input_file in file_list:
            num = extract_number_from_filename(input_file)
            expected_output = f"{num}.obj"

            assert os.path.exists(os.path.join(output_dir, expected_output)), \
                f"Expected output file {expected_output} not found for {input_file}."
            

    ## External tests ##

    # Testing existing count counts
    @pytest.mark.parametrize(*completed_paths)
    def test_modified_file_counts(self, input_dir, output_dir, count_files):
        if not (os.path.exists(input_dir) and os.path.exists(output_dir)):
            assert False, f"Input or output directory does not exist, skipping test."

        npz_count = count_files(input_dir, '.npz')
        obj_count = count_files(output_dir, '.obj')

        assert obj_count >= npz_count, f"Missing obj's, must regenerate. obj(s): {obj_count}, npz(s): {npz_count}"
        assert npz_count >= obj_count, f"Less npz's found than obj files: Either 1) Incorrect directories used or 2) npz's were removed. npz(s): {npz_count}, obj(s): {obj_count}"
        assert npz_count > 0 and obj_count > 0, "Missing files in input or output paths."


    # Testing existing file names
    @pytest.mark.parametrize(*completed_paths)
    def test_existing_filenames(self, input_dir, output_dir, extract_number_from_filename):
        if not (os.path.exists(input_dir) and os.path.exists(output_dir)):
            assert False, f"Input or output directory does not exist, skipping test."
        
        input_files = [f for f in os.listdir(input_dir) if (f.endswith('.npz') and re.search(r'\d', f))]

        for input_file in input_files:
            num = extract_number_from_filename(input_file)
            expected_output = f"{num}.obj"

            assert os.path.exists(os.path.join(output_dir, expected_output)), \
                f"Expected output file {expected_output} not found for {input_file}."
