# Variables {"username":{"type":"str","description":"The username for the user to login inside the database.","regex":"^.*$"},"password":{"type":"secret","description":"The password for the user to login inside the database."},"host":{"type":"str","description":"The host address where the database resides.","regex":"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}$"},"port":{"type":"int","description":"The port on which the database runs.","range":[0,65535]},"database":{"type":"str","description":"The name of the database.","regex":"^.*$"},"table":{"type":"str","description":"The name of the table to load datafrom.","regex":"^.*$"},"target_column":{"type":"str","description":"The name of the target column exactly the same as it is in the databse so this CSV can be used by data scientists.","regex":"^.*$"}}

import pandas as pd
from sqlalchemy import create_engine

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from a PostgreSQL database.

    Args:
    - kwargs should include 'username', 'password', 'host', 'port', 'database', and 'table'.

    Returns:
        pandas.DataFrame - Data loaded from the specified PostgreSQL table.
    """

    username = kwargs.get('username')
    password = kwargs.get('password')
    host = kwargs.get('host')
    port = kwargs.get('port')
    database = kwargs.get('database')
    table = kwargs.get('table')

    if None in [username, password, host, port, database, table]:
        raise ValueError("All connection parameters (username, password, host, port, database, table) must be provided.")


    connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

    engine = create_engine(connection_string)

    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output is not a DataFrame'
