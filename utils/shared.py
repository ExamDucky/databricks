import requests
from config import DATABRICKS_URL, TOKEN
import base64
from fastapi import HTTPException
import os
import requests
import json

def databricks_api(endpoint: str, method: str = "get", data=None):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    url = f"{DATABRICKS_URL}{endpoint}"
    response = requests.request(method, url, headers=headers, json=data)
    return response

def upload_to_databricks(filepath: str, dbfs_path: str):
    with open(filepath, "rb") as f:
        file_content = base64.b64encode(f.read()).decode("utf-8")
    
    response = databricks_api("/api/2.0/dbfs/put", "post", {
        "path": dbfs_path,
        "contents": file_content,
        "overwrite": True
    })
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to upload file to Databricks")
    return dbfs_path

os.environ["DATABRICKS_OAUTH_TOKEN"] = "eyJraWQiOiJkZmJjOWVmMThjZTQ2ZTlhMDg2NWZmYzlkODkxYzJmMjg2NmFjMDM3MWZiNDlmOTdhMDg1MzBjNWYyODU3ZTg4IiwidHlwIjoiYXQrand0IiwiYWxnIjoiUlMyNTYifQ.eyJjbGllbnRfaWQiOiJkYXRhYnJpY2tzLXNlc3Npb24iLCJzY29wZSI6ImFsbC1hcGlzIiwiYXV0aG9yaXphdGlvbl9kZXRhaWxzIjpbeyJvYmplY3RfdHlwZSI6InNlcnZpbmctZW5kcG9pbnRzIiwib2JqZWN0X3BhdGgiOiIvc2VydmluZy1lbmRwb2ludHMvOWM1MzhlNjk5ODI1NDcxODllMjdlNTg1ZTY2MGRiMTUiLCJhY3Rpb25zIjpbInF1ZXJ5X2luZmVyZW5jZV9lbmRwb2ludCJdLCJhbm5vdGF0aW9ucyI6eyJlbmRwb2ludE5hbWUiOiJDb2RlQmVydF9WNiJ9LCJ0eXBlIjoid29ya3NwYWNlX3Blcm1pc3Npb24ifV0sImlzcyI6Imh0dHBzOi8vZGJjLTQzZmM4ODk1LTJiZGIuY2xvdWQuZGF0YWJyaWNrcy5jb20vb2lkYyIsImF1ZCI6IjM1NDc5MDU0OTEwODQwMiIsInN1YiI6Imlka2VpdGhlcjMzMzNAZ21haWwuY29tIiwiaWF0IjoxNzMxMjMyODI5LCJleHAiOjE3MzEyMzY0MjksImp0aSI6ImZlZjFlZmFlLTQ3MjktNDIzZS05MzYwLWJmMDE0MDdhZGVjNCJ9.jZ2mJ_YjAZo8MpfdJMLkm9QvETnCmhz1KIT1ifOvWykjBM0YKPj0wRsGvN9nvRwgMFQLtCZ0d9t9BYi4Rs6Jj6KjIbxGTXORXJDSpeouTgQT2AWKTesssl1L48dsgjqL4lGo4wtenRfWIQdrCCpa1Fa7Oc9TASsOFiMO4-dQJmeAiZb39T8DJUOmrCGywljpjqNOCKJysllvohWrs3pZHdt_L8ykXvUjIP272dNk0ZCBJoNDgVyBoKWUUxquJLCgK5Gcz8z-rsAA48AB0fxAB0WA81UQgGJxZAvH9IgpI9JvOnGA23faG0MnnYyo5Joyw9b7pRPck7Vkq95fD7Rl5g"
def get_embeddings_cloud(code):
    url = 'https://9c538e69982547189e27e585e660db15.serving.cloud.databricks.com/354790549108402/serving-endpoints/CodeBert_V6/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_OAUTH_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'input': code}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()