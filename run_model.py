import chunk
import os
import numpy as np

from tifffile import TiffFile
from skimage.util import view_as_windows

np.random.seed(27)
rng = np.random.default_rng()


def raster_to_numpy_array(filename):
    return np.array(TiffFile(filename).asarray())


class ImageryLoader:
    def __init__(self, filename):
        self.data = raster_to_numpy_array(filename)
        self.number_of_bands = self.data.shape[:3]

    def remove_missing_data(self):
        # A function that takes a row or column of band values (n, 5) and returns the non-empty indices
        def get_non_empty_indices(array):
            return np.where(np.any(array > 0, axis=1))[0]

        # First non-empty (0,0,0,0,0) pixel in the first row
        self.start_pixel_column = get_non_empty_indices(self.data[0])[0]

        # Last non-empty pixel in the last row
        self.end_pixel_column = get_non_empty_indices(self.data[-1])[-1]

        # First non-empty pixel in the end_pixel_column
        self.start_pixel_row = get_non_empty_indices(
            self.data[:, self.end_pixel_column]
        )[0]

        # Last non-empty pixel in the start_pixel_column
        self.end_pixel_row = get_non_empty_indices(
            self.data[:, self.start_pixel_column]
        )[-1]

        # Remove missing data
        self.data = self.conform_data_to_boundary(self.data)

    def conform_data_to_boundary(self, data):
        if not self.start_pixel_row:
            raise Exception(
                "Missing data has not been removed from this instance. "
                + "It cannot be used to crop another data set."
            )

        return data[
            self.start_pixel_row : self.end_pixel_row,
            self.start_pixel_column : self.end_pixel_column,
        ]

    def chunk_data(self, chunk_width=128, chunk_length=128):
        chunks = np.array(
            [
                view_as_windows(
                    self.data[:, :, band],
                    [chunk_width, chunk_length],
                    [chunk_width // 2, chunk_length // 2],
                )
                for band in range(self.number_of_bands)
            ]
        )

        chunks = np.moveaxis(chunks, 0, -1)

        chunks = chunks.reshape(
            (
                chunks.shape[0] * chunks.shape[1],
                chunk_width,
                chunk_length,
                self.number_of_bands,
            ),
        )

        self.data = chunks
