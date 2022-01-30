from os.path import join as path_join
from datetime import datetime


class Configuration:
    # Directories
    logdir = path_join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
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
