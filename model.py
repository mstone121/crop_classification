from functools import partial

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras.optimizers import Nadam

from keras_tuner.tuners import Hyperband

from Configuration import Configuration


def hypermodel(hp):
    DefaultConv2D = partial(
        Conv2D,
        kernel_size=hp.Int("kernel_size", min_value=1, max_value=5, step=1),
        activation=hp.Choice("activation", values=["elu", "relu"]),
        padding="SAME",
    )

    model = Sequential(
        [
            Input(shape=[Configuration.chunk_width, Configuration.chunk_height, 5]),
            DefaultConv2D(
                filters=hp.Int("layer1_filters", min_value=16, max_value=64, step=16)
            ),
            DefaultConv2D(
                filters=hp.Int("layer2_filters", min_value=32, max_value=128, step=32)
            ),
            Dense(units=len(Configuration.crops), activation="softmax"),
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


tuner = Hyperband(
    hypermodel,
    objective="val_accuracy",
    max_epochs=Configuration.max_epochs,
    factor=Configuration.factor,
    directory=Configuration.model_dir,
    project_name=Configuration.hypermodel_name,
)
