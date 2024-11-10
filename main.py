from fastapi import FastAPI, status, Depends
from routers import access, plagiarism
from models.models import HealthCheck
from config import Settings, get_settings
from functools import lru_cache
from typing import Annotated

app = FastAPI()
app.include_router(access.router)
app.include_router(plagiarism.router)

@app.get("/health_check", 
         response_model=HealthCheck, 
         status_code=status.HTTP_200_OK,
         tags=["healthcheck"])
async def health_check() -> HealthCheck:
    return HealthCheck(status="OK")

@app.get("/info")
async def info(settings: Annotated[Settings, Depends(get_settings)]):
    return {
        "DATABRICKS_URL": settings.DATABRICKS_URL,
        "TOKEN": settings.TOKEN,
        "DBFS_ROOT_PATH": settings.DBFS_ROOT_PATH,
    }