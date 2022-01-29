import numpy as np
from tifffile import TiffFile

from DataSet import Boundary, DataSet


class Imagery(DataSet):
    def __init__(self, filename):
        self.data = np.array(TiffFile(filename).asarray())
        super().__init__()

    def remove_missing_data(self) -> Boundary:
        boundary = self.get_data_boundary()
        self.resize_data_set(boundary)
        return boundary

    def normalize(self):
        self.data = self.data / 2 ** 16
