import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
from glob import glob
import tifffile as tf # for 16 bit tiff
import os


# build a tif class with similar interface
class TifStack:
    '''We need a tif stack with an interface that will load a slice one at a time
    We assume each tif has the same size
    We assume 16
    '''

    def __init__(self, input_directory, pattern='*.tif'):
        self.input_directory = input_directory
        self.pattern = pattern
        self.files = glob(os.path.join(input_directory, pattern))
        self.files.sort()
        test = Image.open(self.files[0])
        self.nxy = test.size
        test.close()
        self.nz = len(self.files)
        self.shape = (self.nz, self.nxy[1], self.nxy[0])  # note, it is xy not rowcol

    def __getitem__(self, i):
        return tf.imread(self.files[i]) / (2 ** 16 - 1)

    def __len__(self):
        return len(self.files)

    def close(self):
        pass  # nothing necessary

