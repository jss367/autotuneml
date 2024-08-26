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
        "class_weight": [None, "balanced"],  # for classification
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
