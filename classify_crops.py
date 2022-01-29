from os.path import join as path_join
from datetime import datetime
from numpy import random as np_random
from functools import partial

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import save_model, load_model
from keras_tuner.tuners import Hyperband

from Imagery import Imagery
from Labels import Labels

logdir = path_join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))


def log(message):
    # I might want to write this output to a file at some point
    print(message)


def data_file(filename):
    return path_join("data", filename)


crops = ["Corn", "Soybeans"]

log("Loading data...")
imagery = Imagery(data_file("20130824_RE3_3A_Analytic_Champaign_north.tif"))
labels = Labels(
    data_file("CDL_2013_Champaign_north.tif"),
    data_file("CDL_2013_clip_20170525181724_1012622514.tif.vat.dbf"),
    ["Corn", "Soybeans"],
)

log("Removing unused labels...")
labels.remove_unused_labels()

log("Removing areas with missing data...")
boundary = imagery.get_boundary()
imagery.resize(boundary)
labels.resize(boundary)

log("Chunking Data...")
chunk_width = 128
chunk_height = 128

imagery.chunk(chunk_width, chunk_height)
labels.chunk(chunk_width, chunk_height)

log("Shuffling Data...")
np_random.seed(27)
random_indicies = np_random.permutation(imagery.get_chunk_count())
imagery.shuffle(random_indicies)
labels.shuffle(random_indicies)

log("Normalizing imagery...")
imagery.normalize()

log("Data preparation done. Defering output to Keras...")
log("")

# Define model instance
def hypermodel(hp):
    DefaultConv2D = partial(
        Conv2D,
        kernel_size=hp.Int("kernel_size", min_value=1, max_value=5, step=1),
        activation=hp.Choice("activation", values=["elu", "relu"]),
        padding="SAME",
    )

    model = Sequential(
        [
            Input(shape=[chunk_width, chunk_height, 5]),
            DefaultConv2D(
                filters=hp.Int("layer1_filters", min_value=16, max_value=64, step=16)
            ),
            DefaultConv2D(
                filters=hp.Int("layer2_filters", min_value=32, max_value=128, step=32)
            ),
            Dense(units=len(crops), activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Nadam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        metrics=["accuracy"],
    )

    return model


# Hyper model
tuner = Hyperband(
    hypermodel,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="models",
    project_name="hypermodel_1",
)

# Run search
tuner.search(
    imagery.get_data(),
    labels.get_data(),
    epochs=25,
    validation_split=0.2,
    callbacks=[
        TensorBoard(logdir, histogram_freq=1),
        EarlyStopping(monitor="val_loss", patience=5),
    ],
)
