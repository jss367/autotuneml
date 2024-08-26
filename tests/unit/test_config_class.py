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


@pytest.mark.skip(reason="Skipping this test for now")
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


@pytest.fixture
def temp_multi_model_config_file(tmp_path):
    content = """
class XGBoostConfig:
    hyperopt_space = {
        "max_depth": [3, 18],
        "gamma": [1, 9],
        "reg_alpha": [40, 180],
        "reg_lambda": [0, 1],
        "colsample_bytree": [0.5, 1],
        "min_child_weight": [0, 10],
        "seed": 0,
        "n_estimators": [100, 1000],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.5, 1],
    }

class RandomForestConfig:
    hyperopt_space = {
        "n_estimators": [10, 700],
        "max_depth": [1, 100],
        "min_samples_split": [2, 20],
        "min_samples_leaf": [1, 10],
        "max_features": ["sqrt", "log2"],
        "random_state": 42,
        "criterion": ["gini", "entropy"],
        "min_weight_fraction_leaf": [0, 0.5],
        "bootstrap": [False, True],
    }

class LinearConfig:
    hyperopt_space = {
        "C": [1e-4, 1e4],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [100, 1000],
        "class_weight": [None, "balanced"],
        "fit_intercept": [False, True],
    }

class FastaiTabularConfig:
    hyperopt_space = {
        "n_layers": [1, 5],
        "layer_size": [8, 64],
        "ps": [0, 0.5],
        "bs": [16, 32, 64, 128, 256],
        "lr": [1e-5, 1e-1],
        "embed_p": [0, 0.5],
        "epochs": [3, 20],
        "weight_decay": [1e-5, 1e-1],
    }
    """
    config_file = tmp_path / "multi_model_config.py"
    config_file.write_text(content)
    return str(config_file)


def test_load_multi_model_config(temp_multi_model_config_file):
    config = load_config(temp_multi_model_config_file)

    assert isinstance(config, Config)

    # Check XGBoostConfig
    assert hasattr(config, 'XGBoostConfig_hyperopt_space')
    assert config.XGBoostConfig_hyperopt_space['max_depth'] == [3, 18]
    assert config.XGBoostConfig_hyperopt_space['gamma'] == [1, 9]
    assert config.XGBoostConfig_hyperopt_space['seed'] == 0

    # Check RandomForestConfig
    assert hasattr(config, 'RandomForestConfig_hyperopt_space')
    assert config.RandomForestConfig_hyperopt_space['n_estimators'] == [10, 700]
    assert config.RandomForestConfig_hyperopt_space['max_features'] == ["sqrt", "log2"]
    assert config.RandomForestConfig_hyperopt_space['random_state'] == 42

    # Check LinearConfig
    assert hasattr(config, 'LinearConfig_hyperopt_space')
    assert config.LinearConfig_hyperopt_space['C'] == [1e-4, 1e4]
    assert config.LinearConfig_hyperopt_space['penalty'] == ["l1", "l2"]
    assert config.LinearConfig_hyperopt_space['class_weight'] == [None, "balanced"]

    # Check FastaiTabularConfig
    assert hasattr(config, 'FastaiTabularConfig_hyperopt_space')
    assert config.FastaiTabularConfig_hyperopt_space['n_layers'] == [1, 5]
    assert config.FastaiTabularConfig_hyperopt_space['bs'] == [16, 32, 64, 128, 256]
    assert config.FastaiTabularConfig_hyperopt_space['lr'] == [1e-5, 1e-1]

    # Check that no extra attributes are present
    expected_attributes = {
        'XGBoostConfig_hyperopt_space',
        'RandomForestConfig_hyperopt_space',
        'LinearConfig_hyperopt_space',
        'FastaiTabularConfig_hyperopt_space',
    }
    actual_attributes = set(vars(config).keys())
    assert actual_attributes == expected_attributes, f"Unexpected attributes: {actual_attributes - expected_attributes}"


if __name__ == "__main__":
    pytest.main()
