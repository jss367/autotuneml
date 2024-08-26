import os

import pytest

from autotuneml.configs.config_class import Config, load_config


@pytest.fixture
def temp_config_file(tmp_path):
    def _create_file(content):
        config_file = tmp_path / "test_config.py"
        config_file.write_text(content)
        return str(config_file)

    return _create_file


def test_single_config_class(temp_config_file):
    content = """
class TestConfig:
    param1 = 'value1'
    param2 = 42
    """
    file_path = temp_config_file(content)
    config = load_config(file_path)
    assert isinstance(config, Config)
    assert config.param1 == 'value1'
    assert config.param2 == 42


def test_multiple_config_classes(temp_config_file):
    content = """
class Config1:
    param1 = 'value1'

class Config2:
    param2 = 42

class Config3:
    param3 = [1, 2, 3]
    """
    file_path = temp_config_file(content)
    config = load_config(file_path)
    assert isinstance(config, Config)
    assert config.param1 == 'value1'
    assert config.param2 == 42
    assert config.param3 == [1, 2, 3]


def test_nested_config(temp_config_file):
    content = """
class OuterConfig:
    class InnerConfig:
        inner_param = 'inner_value'
    outer_param = 'outer_value'
    """
    file_path = temp_config_file(content)
    config = load_config(file_path)
    assert isinstance(config, Config)
    assert config.outer_param == 'outer_value'
    assert isinstance(config.InnerConfig, Config)
    assert config.InnerConfig.inner_param == 'inner_value'


def test_config_with_methods(temp_config_file):
    content = """
class ConfigWithMethods:
    param = 'value'
    def method(self):
        return 'method_result'
    """
    file_path = temp_config_file(content)
    config = load_config(file_path)
    assert isinstance(config, Config)
    assert config.param == 'value'
    assert not hasattr(config, 'method')


def test_empty_config_file(temp_config_file):
    content = ""
    file_path = temp_config_file(content)
    with pytest.raises(ValueError, match="No configuration classes found in"):
        load_config(file_path)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config('non_existent_file.py')


def test_config_with_imports(temp_config_file):
    content = """
import os

class ConfigWithImports:
    param = os.path.join('path', 'to', 'file')
    """
    file_path = temp_config_file(content)
    config = load_config(file_path)
    assert isinstance(config, Config)
    assert config.param == os.path.join('path', 'to', 'file')


def test_config_with_complex_types(temp_config_file):
    content = """
from typing import List, Dict

class ComplexConfig:
    list_param: List[int] = [1, 2, 3]
    dict_param: Dict[str, float] = {'a': 1.0, 'b': 2.0}
    """
    file_path = temp_config_file(content)
    config = load_config(file_path)
    assert isinstance(config, Config)
    assert config.list_param == [1, 2, 3]
    assert config.dict_param == {'a': 1.0, 'b': 2.0}


if __name__ == "__main__":
    pytest.main()
