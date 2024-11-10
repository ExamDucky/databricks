import base64
import requests
# from config import DATABRICKS_URL, TOKEN
# Configuration
DATABRICKS_URL = "https://dbc-43fc8895-2bdb.cloud.databricks.com"
TOKEN = ""
DBFS_ROOT_PATH = "dbfs:/model/"
# Helper function to upload a file to DBFS
def upload_to_databricks(filepath, dbfs_path):
    with open(filepath, "rb") as f:
        file_content = base64.b64encode(f.read()).decode("utf-8")
    
    response = requests.post(
        f"{DATABRICKS_URL}/api/2.0/dbfs/put",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"path": dbfs_path, "contents": file_content, "overwrite": True}
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to upload {filepath} to DBFS")
    print(f"File uploaded to {dbfs_path}")
    
# Upload model files to DBFS
upload_to_databricks("./graphcodebert/pytorch_model.bin", DBFS_ROOT_PATH + "pytorch_model.bin")
upload_to_databricks("./graphcodebert/config.json", DBFS_ROOT_PATH + "config.json")
upload_to_databricks("./graphcodebert/vocab.json", DBFS_ROOT_PATH + "vocab.json")
