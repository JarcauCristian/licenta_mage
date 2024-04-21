# Variables {"threshold":{"type":"int","description":"The threshold were lower then this outlier from a column will be removed.","range":[0,5]}}

import numpy as np
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    threshold = kwargs.get("threshold")

    if threshold is None:
        raise ValueError("Threshold kwarg must not be None.")

    numeric_cols = data.select_dtypes(include=[np.number])
    z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
    
    mask = (z_scores > -float(threshold)) & (z_scores < float(threshold))
    filtered_data = data[mask.all(axis=1)]

    return filtered_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
