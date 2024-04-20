# Variables: {"threshold":{"type":"int","description":"The threshold were lower then this columns will be deleted.","range":[0,1]}}

import numpy as np
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def is_null_like(value):
    """
    Check if the provided value is one of the common representations of 'null' in Python,
    including NumPy's NaN and strings that resemble 'NaN'.

    Args:
    value (any): The value to check.

    Returns:
    bool: True if the value is considered 'null-like', False otherwise.
    """

    null_like_values = (None, '', 0, [])

    if value in null_like_values:
        return True

    if isinstance(value, float) and np.isnan(value):
        return True

    if isinstance(value, str) and value.strip().lower() == 'nan':
        return True

    if isinstance(value, str) and value.strip().lower() == 'na':
        return True

    return False



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
        raise TypeError("Input data can only be a pandas DataFrame!")

    threshold = kwargs.get("threshold")

    if threshold is None:
        raise ValueError("Threshold kwargs is mandatory!")

    null_like_percent = data.applymap(lambda x: is_null_like(x)).mean(axis=1)

    rows_to_drop = null_like_percent[null_like_percent > float(threshold)].index

    data.drop(index=rows_to_drop, inplace=True)

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
