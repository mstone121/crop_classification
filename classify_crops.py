from numpy import random as np_random
from os.path import join as path_join

from Imagery import Imagery
from Labels import Labels


def data_file(filename):
    return path_join("data", filename)


imagery = Imagery(data_file("20130824_RE3_3A_Analytic_Champaign_north.tif"))
labels = Labels(
    data_file("CDL_2013_Champaign_north.tif"),
    data_file("CDL_2013_clip_20170525181724_1012622514.tif.vat.dbf"),
    ["Corn", "Soybeans"],
)

# Remove everything but corn and soybean
labels.remove_unused_labels()

# Remove edges of imagery that have no data
boundary = imagery.get_data_boundary()
imagery.resize_data_set(boundary)
labels.resize_data_set(boundary)

# Chunk Data
chunk_width = 128
chunk_height = 128

imagery.chunk(chunk_width, chunk_height)
labels.chunk(chunk_width, chunk_height)

# Shuffle Data
np_random.seed(27)
random_indicies = np_random.permutation(imagery.get_chunk_count())
imagery.shuffle(random_indicies)
labels.shuffle(random_indicies)

# Normalize imagery
imagery.normalize()

# Split Data
split_ratio = 0.8
[training_X, validation_X] = imagery.split(split_ratio)
[training_Y, validation_Y] = labels.split(split_ratio)
