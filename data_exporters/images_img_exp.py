# Variables: {"name":{"type":"str","description":"The name of the file that will be saved in MinIO.","regex":"^[a-z_]+$"},"description":{"type":"str","description":"The description of the CSV file that will be saved inside MinIO."},"category":{"type":"drop_down","description":"The category for Neo4j semnantics.","values":[]},"new_category":{"type":"str","description":"New category name to be created in Neo4j.","regex":"^[a-z_]+$"}, "share_data": {"type": "bool", "description": "Check if you want to share your data with others or not!"}, "dataset_type": {"type": "drop_down", "description": "Types of what tha dataset can be in the case of Machine Learning models.", "values": ["classification", "regression", "clustering"]}}

import io
import os
import json
import yaml
import requests
from datetime import datetime
from mage_ai.settings.repo import get_repo_path

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    
    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    minio_api = config["default"]["MINIO_API"]
    neo4j_api = config["default"]["NEO4J_API"]


    description = kwargs.get("description")
    category = kwargs.get("category")
    new_category = kwargs.get("new_category")

    final_category = str(category).lower() if new_category is None else str(new_category).lower()

    headers = {
        "Authorization": "Bearer mage_7dfe20e1136346b9dc5254f8e1941f97224a8bb50bb4b28d6acc3932a98600ec",
        "Content-type": "application/json"
    }

    if new_category is not None:
        response = requests.post(f"{neo4j_api}/category/create?name={final_category}", headers=headers)
        if response.status_code == 201:
            print("Category Created Successfully!")
        else:
            raise ValueError("Did not receive satisfactory response from API.")

    name = kwargs.get("name")

    if kwargs.get("column_descriptions") is None:
        column_descriptions = {}
    else:
        column_descriptions = kwargs.get("column_descriptions") if isinstance(kwargs.get("column_descriptions"), dict) else json.loads(kwargs.get("column_descriptions"))

    column_descriptions["share_data"] = "true" if kwargs.get("share_data").lower() == "on" else "false"
    column_descriptions["dataset_type"] = str(kwargs.get("dataset_type"))
    column_descriptions["last_accessed"] = str(datetime.now())
    column_descriptions["target_column"] = str(kwargs.get("target_column"))

    csv_data = data.to_csv(index=False)
    files = {
        'file': ('filename.csv', io.BytesIO(csv_data.encode('utf-8'))),
    }

    payload = {
        'name': name,
        'tags': json.dumps({}),
        'temporary': 'false'
    }

    response = requests.post(f"{minio_api}/upload_free", files=files, data=payload)

    if response.status_code == 201:
        neo4j_payload = {
            "name": str(name.split("/")[-1]),
            "belongs_to": str(final_category),
            "url": str(response.json()),
            "tags": column_descriptions,
            "user": str(name.split("/")[0]),
            "description": description
        }

        response = requests.post(f"{neo4j_api}/dataset/create", json=neo4j_payload, headers=headers)

        if response.status_code != 201:
            raise ValueError("Dataset Could Not Be Created!")
    else:
        raise ValueError(f"Error: {response.status_code} {response.content.decode('utf-8')}")
