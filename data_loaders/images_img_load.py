# Variables: {"initial_name":{"type":"str","description":"The name of the saved file in the temp bucket.","regex":"^[a-z_]+$"},"file":{"type":"file","description":"The CSV dataset that will be used for this pipeline."}}

import os
import io
import yaml
import requests
import pandas as pd
from mage_ai.settings.repo import get_repo_path

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def config_loader():
    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    initial_name = kwargs.get("initial_name")

    if initial_name is None:
        raise ValueError("Kwarg initial_name required!")

    config = config_loader()

    response = requests.get(f'{config["default"]["MINIO_API"]}/get_object?dataset_path={initial_name}&forever=false')

    if response.status_code != 200:
        raise ValueError('Failed to get information about the saved dataset.')

    data_response = requests.get(response.json()["url"])

    if data_response.status_code != 200:
        raise ValueError('Failed to get the dataset.')

    df = pd.read_csv(io.BytesIO(data_response.content))

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
