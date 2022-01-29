from itertools import permutations
import numpy as np

from skimage.util import view_as_windows


class Boundary:
    def __init__(self, start_column, end_column, start_row, end_row):
        self.start_column = start_column
        self.end_column = end_column
        self.start_row = start_row
        self.end_row = end_row


class DataSet:
    def __init__(self) -> None:
        if not self.data:
            raise Exception(
                "DataSet is an abstract class. self.data must be set before initialization."
            )

        shape = self.data.shape
        if shape > 3 or shape < 2:
            raise Exception("Invalid data shape")

        self.number_of_bands = 1 if len(shape) == 2 else shape[2]

    def resize(self, boundary: Boundary):
        self.data = self.data[
            boundary.start_row : boundary.end_row,
            boundary.start_column,
            boundary.end_column,
        ]

    def get_boundary(self) -> Boundary:
        # A function that takes a row or column of band values (n, 5) and returns the non-empty indices
        def get_non_empty_indices(array):
            return np.where(np.any(array > 0, axis=1))[0]

        # First non-empty (0,0,0,0,0) pixel in the first row
        start_column = get_non_empty_indices(self.data[0])[0]

        # Last non-empty pixel in the last row
        end_column = get_non_empty_indices(self.data[-1])[-1]

        # First non-empty pixel in the end_column
        start_row = get_non_empty_indices(self.data[:, end_column])[0]

        # Last non-empty pixel in the start_column
        end_row = get_non_empty_indices(self.data[:, start_column])[-1]

        # Remove missing data
        return Boundary(start_column, end_column, start_row, end_row)

    def chunk(self, chunk_width=128, chunk_length=128):
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
        self.chunk_count = chunks.shape[0]

    def get_chunk_count(self):
        return self.chunk_count

    def shuffle(self, shuffle_indicies: permutations):
        self.data = self.data[shuffle_indicies]

    def split(self, split_ratio):
        split_point = int(len(self.data) * split_ratio)

        return [self.data[:split_point], self.data[:split_point]]
