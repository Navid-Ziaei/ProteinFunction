import json
import datetime
from pathlib import Path
import os


class Paths:
    """
    A class to manage paths for a given set of settings.

    Attributes:
    - num_folds: Number of folds for cross-validation.
    - folder_name: Name of the folder to store results.
    - path_result: List of paths to store results.
    - path_model: List of paths to store models.
    - raw_dataset: Path to the raw dataset.
    - feature_path: Path to the features.
    - models: Path to the models.
    - debug_mode: Boolean indicating if the debug mode is active.
    """

    def __init__(self, settings):
        """Initialize the Paths class with given settings."""
        self.num_folds = settings.num_fold
        self.folder_name = None
        self.path_result = []
        self.path_model = []
        self.raw_dataset = './'
        self.feature_path = './'
        self.model = './'
        self.debug_mode = settings.debug_mode

    def load_device_paths(self):
        """
        Load device paths from a JSON configuration file.
        """

        # Get the working directory
        working_folder = os.path.dirname(os.path.realpath(__file__))
        working_folder = os.path.dirname(os.path.dirname(working_folder))

        # Load device path from the JSON file
        try:
            with open(working_folder + "/configs/device_path.json", "r") as file:
                device = json.loads(file.read())
        except:
            raise Exception('Could not load device_path.json from the working directory!')

        # Set the paths from the JSON file to the class attributes
        for key, value in device.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))

        self.create_paths()

    def create_paths(self):
        """
        Create paths for results and models based on the given settings.
        """

        # Get the working directory path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))

        # Set the base path for results
        self.base_path = dir_path + '\\results\\'

        # Set the folder name based on the debug mode
        if self.debug_mode is False:
            self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.folder_name = 'debug'

        results_base_path = self.base_path + '/' + self.folder_name + '/'

        # Create directories for results and models
        Path(results_base_path).mkdir(parents=True, exist_ok=True)
        self.path_model.append(os.path.join(results_base_path + '/models/'))
        self.path_result.append(os.path.join(results_base_path + '/'))

        # Create directories for each fold if num_folds is greater than 1
        if self.num_folds > 1:
            for i in range(self.num_folds):
                Path(results_base_path + '/fold{}/models'.format(i + 1)).mkdir(parents=True, exist_ok=True)
                Path(results_base_path + '/fold{}'.format(i + 1)).mkdir(parents=True, exist_ok=True)
                self.path_model.append(os.path.join(results_base_path + '/fold{}/models/'.format(i + 1)))
                self.path_result.append(os.path.join(results_base_path + '/fold{}/'.format(i + 1)))
