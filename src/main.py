from src.settings import Settings, Paths
from src.data_loader import DataLoader
from src.models import BaseLineMethod
from src.evaluation import ResultGenerator


# load settings from settings.json
settings = Settings()
settings.load_settings()

# load device paths from device_path.json and create result paths
paths = Paths(settings)
paths.load_device_paths()

# load raw data
data_loader = DataLoader(paths, settings)
data_loader.import_data()

# train models
model = BaseLineMethod(settings, paths, data_loader)
model.train()

# evaluation
evaluator = ResultGenerator(model, data_loader, settings, paths)
evaluator.get_output_csv()