import importlib.util
import inspect
import os
from types import SimpleNamespace


class Config(SimpleNamespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

    def items(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}.items()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data):
        def convert(obj):
            if isinstance(obj, dict):
                return cls(**{k: convert(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        return convert(data)


def convert_to_config(cls):
    return Config(
        **{key: getattr(cls, key) for key in dir(cls) if not key.startswith("__") and not callable(getattr(cls, key))}
    )


def load_config(file_path: str) -> Config:
    """
    This loads all configuration classes from a Python file.
    Returns a composite Config object with nested Config objects for each class.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Configuration file {file_path} not found.")

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    composite_config = {}
    for name, obj in inspect.getmembers(config_module, inspect.isclass):
        if obj.__module__ == module_name:
            class_config = convert_to_config(obj)
            composite_config[name] = class_config

    if not composite_config:
        raise ValueError(f"No configuration classes found in {file_path}")

    return Config(**composite_config)
