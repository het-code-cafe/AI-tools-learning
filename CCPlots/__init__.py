import importlib
import os


def load_classes():
    """ Load the different classes, ignore the utilities and execute the demos """
    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "config.py", "utils.py"]:
            module_name = filename[:-3]  # Remove .py extension
            module = importlib.import_module(f'.{module_name}', package='CCPlots')
            for name, cls in module.__dict__.items():
                if isinstance(cls, type):
                    globals()[name] = cls


load_classes()
