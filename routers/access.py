from fastapi import File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from utils.shared import databricks_api, upload_to_databricks
from utils.access import download_file_from_dbfs
import tempfile
import aiofiles
import requests
from config import settings

router = APIRouter(
    prefix="/access",
    tags=['Access']
)

@router.get("/list-folders/")
async def list_folders():
    response = databricks_api("/api/2.0/dbfs/list", "get", {"path": settings.DBFS_ROOT_PATH})  
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to list folders on DBFS")
    
    folders = response.json().get("files", [])
    directory_structure = {}
    
    for folder in folders:
        if folder["is_dir"]:
            person_name = folder["path"].split("/")[-1]
            folder_response = databricks_api("/api/2.0/dbfs/list", "get", {"path": folder["path"]})
            
            if folder_response.status_code == 200:
                files = [f["path"].split("/")[-1] for f in folder_response.json().get("files", [])]
                directory_structure[person_name] = files
            else:
                directory_structure[person_name] = "Failed to retrieve files"
    return JSONResponse(content=directory_structure)


@router.get("/download-file/")
async def download_file(folder_name: str, filename: str, ):
    dbfs_path = f"{settings.DBFS_ROOT_PATH}{folder_name}/{filename}"
    return {"file-content": download_file_from_dbfs(dbfs_path)}


@router.post("/upload-files/")
async def upload_files(test_file: UploadFile = File(...), name_file: UploadFile = File(...), ):
    # Save the files temporarily using tempfile
    temp_test_path = tempfile.NamedTemporaryFile(delete=False).name
    temp_name_path = tempfile.NamedTemporaryFile(delete=False).name 
    name = name_file.filename.split(".")[0]
    DBFS_PATH = f"dbfs:/baza/{name}/"
    headers = {
        "Authorization": f"Bearer {settings.TOKEN}"
    }

    # Create a directory
    response = requests.post(
        f"{settings.DATABRICKS_URL}/api/2.0/dbfs/mkdirs",
        headers=headers,
        json={"path": f"/baza/{name}"}
    )

    if response.status_code == 200:
        print("Directory created successfully.")
    else:
        print("Failed to create directory:", response.text)

    async with aiofiles.open(temp_test_path, 'wb') as out_file:
        content = await test_file.read()
        await out_file.write(content)
    
    async with aiofiles.open(temp_name_path, 'wb') as out_file:
        content = await name_file.read()
        await out_file.write(content)

    upload_to_databricks(temp_test_path, DBFS_PATH + test_file.filename)
    upload_to_databricks(temp_name_path, DBFS_PATH + name_file.filename)

    return {"message": "Files uploaded successfully"}

@router.delete("/delete-all-folders/")
def delete_folders():
    try:
        data = {
                "path": settings.DBFS_ROOT_PATH,
                "recursive": True  # Set to True if deleting a directory
            }
            
        # Send delete request to Databricks
        databricks_api("/api/2.0/dbfs/delete", "post", data)
        headers = {
        "Authorization": f"Bearer {settings.TOKEN}"
        }

        # Create a directory
        response = requests.post(
            f"{settings.DATABRICKS_URL}/api/2.0/dbfs/mkdirs",
            headers=headers,
            json={"path": f"/baza/"}
        )

        if response.status_code == 200:
            print("Directory created successfully.")
        else:
            print("Failed to create directory:", response.text)
            return {"status": "success", "message": f"File '{settings.DBFS_ROOT_PATH}' deleted successfully"}

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"Failed to delete file: {e.detail}")


@router.delete("/delete-folder/")
def delete_folder(folder_name:str, ):
    try:
        data = {
            "path": settings.DBFS_ROOT_PATH + "/" + folder_name,
            "recursive": True  # Set to True if deleting a directory
        }
        
        # Send delete request to Databricks
        databricks_api("/api/2.0/dbfs/delete", "post", data)
        return {"status": "success", "message": f"File '{settings.DBFS_ROOT_PATH + folder_name}' deleted successfully"}
    
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"Failed to delete file: {e.detail}")