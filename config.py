from pydantic_settings import BaseSettings
from functools import lru_cache

DATABRICKS_URL = "https://dbc-43fc8895-2bdb.cloud.databricks.com"
TOKEN = ""
DBFS_ROOT_PATH = "dbfs:/baza/"

class Settings(BaseSettings):
    DATABRICKS_URL: str
    TOKEN: str
    DBFS_ROOT_PATH: str
    EXPERIMENT_URL: str
    model1_name: str
    model2_name: str
    codebert_url: str
    OAUTH_TOKEN_CODE: str
    graphcodebert_url: str
    OAUTH_TOKEN_GRAPH: str
    DATABRICKS_OAUTH_TOKEN: str
    CODE_BERT_URL: str

    class Config:
        env_file = ".env"  # Specify the path to your .env file

@lru_cache
def get_settings():
    return Settings()

settings = get_settings()