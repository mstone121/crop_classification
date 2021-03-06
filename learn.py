from numpy import random as np_random
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from Configuration import Configuration, log
from Imagery import Imagery
from Labels import Labels
from model import tuner

# The log() method calls here sort of double as comments delineating sections of the process

log("Loading data...")
imagery = Imagery(Configuration.training_imagery_file)
labels = Labels(
    Configuration.training_labels_file,
    Configuration.training_labels_metadata_file,
    Configuration.crops,
)

log("Removing unused labels...")
labels.remove_unused_labels()

log("Removing areas with missing data...")
boundary = imagery.get_boundary()
imagery.resize(boundary)
labels.resize(boundary)

log("Chunking Data...")
imagery.chunk(Configuration.chunk_width, Configuration.chunk_height)
labels.chunk(Configuration.chunk_width, Configuration.chunk_height)

log("Shuffling Data...")
np_random.seed(27)
random_indicies = np_random.permutation(imagery.get_chunk_count())
imagery.shuffle(random_indicies)
labels.shuffle(random_indicies)

log("Normalizing imagery...")
imagery.normalize()

log("Data preparation done. Defering output to Keras...")
log("")

# Run search
tuner.search(
    imagery.get_data(),
    labels.get_data(),
    epochs=Configuration.search_epochs,
    validation_split=Configuration.validation_split,
    callbacks=[
        TensorBoard(Configuration.logdir, histogram_freq=1),
        EarlyStopping(monitor="val_loss", patience=Configuration.patience),
    ],
)
