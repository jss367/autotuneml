import numpy as np
import pytest
from hyperopt import hp
from hyperopt.pyll import Apply, as_apply, scope

from autotuneml.configs.config_class import Config
from autotuneml.skl_train import convert_config_to_hyperopt_space


def is_int_hyperopt_space(space):
    return (
        isinstance(space, Apply)
        and space.name == 'int'
        and isinstance(space.pos_args[0], Apply)
        and space.pos_args[0].name == 'float'
    )


def test_integer_range():
    config = Config(n_layers=[1, 5])
    result = convert_config_to_hyperopt_space(config)
    assert is_int_hyperopt_space(result['n_layers'])
    quniform = result['n_layers'].pos_args[0]
    assert quniform.pos_args[0].pos_args[0].obj == 'n_layers'
    assert quniform.pos_args[1] == 1
    assert quniform.pos_args[2] == 5
    assert quniform.pos_args[3] == 1  # step size


def test_float_range():
    config = Config(learning_rate=[0.001, 0.1])
    result = convert_config_to_hyperopt_space(config)
    assert isinstance(result['learning_rate'], hp.uniform)
    assert result['learning_rate'].pos_args[0] == 'learning_rate'
    assert result['learning_rate'].pos_args[1] == 0.001
    assert result['learning_rate'].pos_args[2] == 0.1


def test_logarithmic_range():
    config = Config(lr=[1e-5, 0.1])
    result = convert_config_to_hyperopt_space(config)
    assert isinstance(result['lr'], hp.loguniform)
    assert result['lr'].pos_args[0] == 'lr'
    assert np.isclose(result['lr'].pos_args[1], np.log(1e-5))
    assert np.isclose(result['lr'].pos_args[2], np.log(0.1))


def test_categorical_choice():
    config = Config(activation=['relu', 'tanh', 'sigmoid'])
    result = convert_config_to_hyperopt_space(config)
    assert isinstance(result['activation'], hp.choice)
    assert result['activation'].pos_args[0] == 'activation'
    assert result['activation'].pos_args[1] == ['relu', 'tanh', 'sigmoid']


def test_single_value():
    config = Config(batch_size=32)
    result = convert_config_to_hyperopt_space(config)
    assert result['batch_size'] == 32


def test_boolean_value():
    config = Config(use_bias=True)
    result = convert_config_to_hyperopt_space(config)
    assert result['use_bias'] is True


def test_mixed_config():
    config = Config(
        n_layers=[1, 5],
        layer_size=[8, 64],
        ps=[0, 0.5],
        bs=[16, 32, 64, 128, 256],
        lr=[1e-5, 0.1],
        embed_p=[0, 0.5],
        epochs=[3, 20],
        weight_decay=[1e-5, 0.1],
        activation='relu',
    )
    result = convert_config_to_hyperopt_space(config)

    assert is_int_hyperopt_space(result['n_layers'])
    assert is_int_hyperopt_space(result['layer_size'])
    assert isinstance(result['ps'], hp.uniform)
    assert isinstance(result['bs'], hp.choice)
    assert isinstance(result['lr'], hp.loguniform)
    assert isinstance(result['embed_p'], hp.uniform)
    assert is_int_hyperopt_space(result['epochs'])
    assert isinstance(result['weight_decay'], hp.loguniform)
    assert result['activation'] == 'relu'


def test_invalid_range():
    config = Config(invalid_range=[5, 1])
    result = convert_config_to_hyperopt_space(config)
    assert is_int_hyperopt_space(result['invalid_range'])
    quniform = result['invalid_range'].pos_args[0]
    assert quniform.pos_args[1] == 1
    assert quniform.pos_args[2] == 5


def test_empty_config():
    config = Config()
    result = convert_config_to_hyperopt_space(config)
    assert result == {}


@pytest.mark.parametrize("value", ["string", None, complex(1, 2)])
def test_unsupported_types(value):
    config = Config(unsupported=value)
    result = convert_config_to_hyperopt_space(config)
    assert isinstance(result['unsupported'], hp.choice)
    assert result['unsupported'].pos_args[1] == [value]


if __name__ == "__main__":
    pytest.main()
