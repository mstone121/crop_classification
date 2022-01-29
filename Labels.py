import numpy as np
import dbfread as DBF

from tifffile import TiffFile
from pandas import DataFrame

from DataSet import DataSet


class Labels(DataSet):
    def __init__(self, filename, metafile, crops) -> None:
        self.data = np.array(TiffFile(filename).asarray())
        self.meta = DataFrame(iter(DBF(metafile)))

        self.crop_map = {
            self.get_crop_value(crops[i])["VALUE"]: i + 1 for i in range(len(crops))
        }

        self.color_map = {crop: self.get_crop_color(crop) for crop in crops}

        super().__init__()

    def get_crop_meta(self, crop):
        return self.meta[self.meta["CLASS_NAME"] == crop]

    def get_crop_color(self, crop):
        return self.get_crop_meta(crop)["RED", "GREEN", "BLUE", "OPACITY"]
