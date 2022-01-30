from os.path import join as path_join
from datetime import datetime


def log(message):
    # I might want to write this output to a file at some point
    print(message)


def data_file(filename):
    return path_join("data", filename)


class Configuration:
    # Directories and files
    training_imagery_file = data_file("20130824_RE3_3A_Analytic_Champaign_north.tif")
    training_labels_file = data_file("CDL_2013_Champaign_north.tif")
    training_labels_metadata_file = data_file(
        "CDL_2013_clip_20170525181724_1012622514.tif.vat.dbf"
    )
    test_imagery_file = data_file("20130824_RE3_3A_Analytic_Champaign_south.tif")

    logdir = path_join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    prediction_file = "predictions"
    model_dir = "models"
    hypermodel_name = "hypermodel_1"

    # Data configuration
    crops = ["Corn", "Soybeans"]
    chunk_width = 128
    chunk_height = 128

    # Hypermodel params
    max_epochs = 10
    factor = 3

    # Search Params
    search_epochs = 25
    validation_split = 0.2
    patience = 5
