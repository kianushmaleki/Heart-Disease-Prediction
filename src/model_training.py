import yaml

# load the model parameters from the YAML file
def load_model_params(config_file: str) -> dict:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


