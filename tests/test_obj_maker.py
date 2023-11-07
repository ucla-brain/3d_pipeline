#! /usr/bin/env python
import pytest
import obj_maker 
import tempfile
import shutil
import os


# Update input paths below for live tests (small folder size recommended)
test_paths = [
    # "/ifshome/cestrada/3d_pipeline/input/",
]

# Update input and output paths for testing existing obj's 
completed_paths = ('input_dir, output_dir', [
#    ("/ifshome/cestrada/3d_pipeline/input/","/ifshome/cestrada/3d_pipeline/out/"),
    #NB1
    ("/panfs/dong/3d_reg/20220818_SW220405_05_LS_6x_1000z/registration_20/20220818_SW220405_05_LS_6x_1000z_ch_1/","/panfs/dong/3d_reg/20220818_SW220405_05_LS_6x_1000z/registration_20/objs_v4/"),

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

    # Testing new obj files generated (both counts and names)
    @pytest.mark.parametrize("input_dir", test_paths)
    def test_create_obj_files(self, input_dir, temp_output_dir, extract_number_from_filename):
        if not (os.path.exists(input_dir)):
            pytest.skip("Input directory does not exist, skipping test.")
        file_list = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.npz')]
        output_dir = temp_output_dir
        obj_maker.create_obj_files(input_dir, output_dir, file_list, 1, None, None, False)
        obj_count = obj_maker.count_files(output_dir, '.obj')

        assert len(obj_maker.generated_obj_files) == obj_count

        for input_file in file_list:
            num = extract_number_from_filename(input_file)
            expected_output = f"{num}.obj"

            assert os.path.exists(os.path.join(output_dir, expected_output)), \
                f"Expected output file {expected_output} not found for {input_file}."


    # Testing existing output counts
    @pytest.mark.parametrize(*completed_paths)
    def test_count_files(self, input_dir, output_dir):
        if not (os.path.exists(input_dir) and os.path.exists(output_dir)):
            pytest.skip("Input or output directory does not exist, skipping test.")

        npz_count = obj_maker.count_files(input_dir, '.npz')
        obj_count = obj_maker.count_files(output_dir, '.obj')

                
        assert obj_count >= npz_count, f"Missing obj's, must regenerate. obj(s): {obj_count}, npz(s): {npz_count}"
        assert npz_count >= obj_count, f"Less npz's found than obj files: Either 1) Incorrect directories used or 2) npz's were removed. npz(s): {npz_count}, obj(s): {obj_count}"
        assert npz_count > 0 and obj_count > 0, "Missing files in input or output paths."


    # Testing existing output file names
    @pytest.mark.parametrize(*completed_paths)
    def test_existing_filenames(self, input_dir, output_dir, extract_number_from_filename):
        if not (os.path.exists(input_dir) and os.path.exists(output_dir)):
            pytest.skip("Input or output directory does not exist, skipping test.")
        
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

        for input_file in input_files:
            num = extract_number_from_filename(input_file)
            expected_output = f"{num}.obj"

            assert os.path.exists(os.path.join(output_dir, expected_output)), \
                f"Expected output file {expected_output} not found for {input_file}."
            
    
        