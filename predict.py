from numpy import argmax, savez_compressed as np_savez_compressed

from Configuration import Configuration, log
from Imagery import Imagery
from model import tuner


# I don't really need the Imagery/Dataset functions for this part.
# I'll use it to load the file and normalize, then get the numpy array.
log("Loading data...")
original_test_data = Imagery(Configuration.test_imagery_file)

log("Normalizing data...")
original_test_data.normalize()
normalized_test_data = original_test_data.get_data()

log("Resizing data to fit model...")
chunk_row_count = normalized_test_data.shape[0] // Configuration.chunk_width
chunk_col_count = normalized_test_data.shape[1] // Configuration.chunk_height

test_data = normalized_test_data[
    : chunk_row_count * Configuration.chunk_width,
    : chunk_col_count * Configuration.chunk_height,
]

log("Chunking data...")
test_data_chunks = test_data.reshape(
    (
        chunk_row_count * chunk_col_count,
        Configuration.chunk_width,
        Configuration.chunk_height,
        test_data.shape[2],
    )
)

log("Predicting...")
parameters = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(parameters)

predictions = model.predict(test_data_chunks)

log("Joining chunks...")
predictions = argmax(predictions, axis=-1)
predictions = predictions.reshape(
    (
        chunk_row_count * Configuration.chunk_width,
        chunk_col_count * Configuration.chunk_height,
    )
)

log("Saving to file...")
np_savez_compressed(
    Configuration.prediction_file,
    predictions,
)
