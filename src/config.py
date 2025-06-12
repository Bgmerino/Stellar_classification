import yaml

def load_config(path):
    """

    Function loads configuration from YAML file.

    :param path: string with path to config file
    :return: dictionary with the configuration parameters
    """

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config