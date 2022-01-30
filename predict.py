from numpy import argmax

from Configuration import Configuration, data_file, log
from Imagery import Imagery
from model import tuner


# I don't really need the Imagery/Dataset functions for this part.
# I'll use it to load the file and normalize, then get the numpy array.
log("Loading data...")
original_test_data = Imagery(data_file("20130824_RE3_3A_Analytic_Champaign_south.tif"))

log("Normalizing data...")
original_test_data.normalize()
normalized_test_data = original_test_data.get_data()

log("Resizing data to fit model...")
chunk_row_count = normalized_test_data.shape[0] // Configuration.chunk_width
chunk_col_count = normalized_test_data.shape[1] // Configuration.chunk_length

test_data = normalized_test_data[
    : chunk_row_count * Configuration.chunk_width,
    : chunk_col_count * Configuration.chunk_length,
]

log("Chunking data...")
test_data_chunks = test_data.reshape(
    (
        chunk_row_count * chunk_col_count,
        Configuration.chunk_width,
        Configuration.chunk_length,
        original_test_data.shape[2],
    )
)

log("Predicting...")
parameters = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(parameters)

predictions = model.predict(test_data_chunks)

log("Joining chunks...")
final_figure = argmax(predictions, axis=-1)
final_figure = final_figure.reshape(
    (
        chunk_row_count * Configuration.chunk_width,
        chunk_col_count * Configuration.chunk_length,
    )
)
