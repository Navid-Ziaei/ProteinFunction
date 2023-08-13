import json
import os


class Settings:
    def __init__(self):
        self.model = 'BaseLine'
        self.__load_feature = None
        self.__save_features = None
        self.__load_pretrained_model = None
        self.__model = None
        self.__debug_mode = False
        self.__num_fold = 5
        self.__test_size = 0.2
        self.__num_of_labels = 1500
        self.__batch_size = None
        self.__epochs = None


    def load_settings(self):
        """
        This function loads the json files for settings and network settings from the working directory and
        creates a Settings object based on the fields in the json file. It also loads the local path of the dataset
        from device_path.json
        return:
            settings: a Settings object
            network_settings: a dictionary containing settings of the models
            device_path: the path to the datasets on the local device
        """

        """ working directory """
        working_folder = os.path.dirname(os.path.realpath(__file__))
        parent_folder = os.path.dirname(os.path.dirname(working_folder)) + '\\'

        """ loading settings from the json file """
        try:
            with open(parent_folder + "/configs/settings.json", "r") as file:
                settings_json = json.loads(file.read())
        except:
            raise Exception('Could not load settings.json from the working directory!')

        """ creating settings """
        if "model" not in settings_json.keys():
            raise Exception('"model" was not found in settings.json!')

        self.model = settings_json["model"]
        del settings_json["model"]

        for key, value in settings_json.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))

    @property
    def epochs(self):
        return self.__epochs
    @epochs.setter
    def epochs(self, value):
        if isinstance(value, int) and value > 0:
            self.__epochs = value
        else:
            raise ValueError("epochs should be an integer number ")

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        if isinstance(value, int) and value > 0:
            self.__batch_size = value
        else:
            raise ValueError("batch_size should be an integer number ")

    @property
    def num_of_labels(self):
        return self.__num_of_labels

    @num_of_labels.setter
    def num_of_labels(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_of_labels = value
        else:
            raise ValueError("num_of_labels should be an integer number ")

    @property
    def num_fold(self):
        return self.__num_fold

    @num_fold.setter
    def num_fold(self, k):
        if isinstance(k, int) and k > 0:
            self.__num_fold = k
        else:
            raise ValueError("num_fold should be integer bigger than 0")

    @property
    def test_size(self):
        return self.__test_size

    @test_size.setter
    def test_size(self, value):
        if 0 < value < 1:
            self.__test_size = value
        else:
            raise ValueError("test_size should be float number between 0 to 1")

    @property
    def load_pretrained_model(self):
        return self.__load_pretrained_model

    @load_pretrained_model.setter
    def load_pretrained_model(self, value):
        if isinstance(value, bool):
            self.__load_pretrained_model = value
        else:
            raise ValueError("load_pretrained_model should be True or False")

    @property
    def load_feature(self):
        return self.__load_feature

    @load_feature.setter
    def load_feature(self, value):
        if isinstance(value, bool):
            self.__load_feature = value
        else:
            raise ValueError("load_feature should be True or False")

    @property
    def save_features(self):
        return self.__save_features

    @save_features.setter
    def save_features(self, value):
        if isinstance(value, bool):
            self.__save_features = value
        else:
            raise ValueError("save_features should be True or False")

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        if isinstance(value, bool):
            self.__debug_mode = value
        else:
            raise ValueError("The v should be boolean")
