class PipelineConfig:
    # Model Selection
    # models = ["xgboost"]
    models = ["random_forest", "linear"]
    models = ["xgboost", "random_forest", "linear"]  # Add "fastai_tabular" if needed

    # Hyperparameter Optimization
    num_trials = 20

    # Data Splitting
    test_size = 0.25
    random_state = 42
    split_method = "random"  # or "date"

    # Problem Type
    problem_type = "classification"  # or "regression"
