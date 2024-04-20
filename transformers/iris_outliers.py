# Variables {"threshold":{"type":"int","description":"The threshold were lower then this outlier from a column will be removed.","range":[0,5]}}

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

    if None in threshold:
        raise ValueError("Threshold kwarg must not be None.")
    
    z_scores = (data - data.mean()) / data.std()
    filtered_data = data[(z_scores > -float(threshold)) & (z_scores < float(threshold))]

    return filtered_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
