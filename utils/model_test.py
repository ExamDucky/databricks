import os
import requests
import numpy as np
import pandas as pd
import json

os.environ["DATABRICKS_OAUTH_TOKEN"] = ""

def score_model():
    url = 'https://7843aea77887431f972197f1311cd6a0.serving.cloud.databricks.com/354790549108402/serving-endpoints/CODEBERT_V2/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_OAUTH_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'input': "int main()"}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


print(score_model())