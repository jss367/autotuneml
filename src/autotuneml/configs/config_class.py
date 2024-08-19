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
