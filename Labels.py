import numpy as np

from tifffile import TiffFile
from dbfread import DBF
from pandas import DataFrame

from DataSet import DataSet


class Labels(DataSet):
    def __init__(self, filename, metafile, crops) -> None:
        self.data = np.array(TiffFile(filename).asarray())
        self.meta = DataFrame(iter(DBF(metafile)))

        self.crop_map = {
            int(self.get_crop_meta(crops[i])["VALUE"]): i + 1 for i in range(len(crops))
        }

        self.color_map = {crop: self.get_crop_color(crop) for crop in crops}

        super().__init__()

    def get_crop_meta(self, crop):
        return self.meta[self.meta["CLASS_NAME"] == crop]

    def get_crop_color(self, crop):
        meta = self.get_crop_meta(crop)
        return (meta["RED"], meta["GREEN"], meta["BLUE"], meta["OPACITY"])

    def remove_unused_labels(self):
        self.data = np.vectorize(lambda value: self.crop_map.get(value, 0))(self.data)
